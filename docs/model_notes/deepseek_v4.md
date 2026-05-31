# deepseek_v4 架构说明

`deepseek_v4` 是本项目对 DeepSeek V4 公开语言模型拓扑的 100M-scale 缩放复现。
它不是官方模型实现，也不复现官方权重、训练数据、后训练流程或工业 kernel。
它的重点是把 DeepSeek V4 的公开结构思想放进本项目统一的 config、model registry、
training、generation 和 benchmark 接口中。

## 公开架构依据

主要依据：

- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4
- NVIDIA Megatron Bridge docs: https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/deepseek/deepseek-v4.html

截至 2026-05-31，可公开核对的关键点包括：

- DeepSeek V4 是 MoE language model。
- attention 从 DeepSeek V3 的 MLA 转向 hybrid local + long-range attention。
- decoder block 由 `layer_types` 选择不同 attention 类型。
- 公开文档包含 `sliding_attention`、`compressed_sparse_attention` 和
  `heavily_compressed_attention` 等概念。
- attention 使用很少 KV heads，公开配置中包含 `num_key_value_heads = 1`。
- 使用 partial RoPE。
- 使用 attention sink、compressed entries、top-k compressed selection 等长程
  attention 相关结构。
- 使用 Manifold-Constrained Hyper-Connections (mHC) 的 multi-stream residual
  思想。
- MoE schedule 中包含 hash-routed MoE 和 routed MoE。
- router scoring 包含 `sqrtsoftplus`。
- 包含 shared experts 和 MTP/next-n prediction 相关接口。

## 定位

这个模型回答的问题是：在约 100M total parameters 下，MoE + hybrid compressed
attention 的结构会带来怎样的 active-parameter、速度和 loss 曲线差异。

它和其他三个模型最大的不同是参数口径：

- `total_parameters` 表示模型所有权重。
- `active_parameters_per_token` 表示每个 token 实际经过的参数近似量。

因此比较 `deepseek_v4` 时不能只看 total parameters。

## 当前正式配置

配置文件：`configs/deepseek_v4.yaml`

```yaml
architecture: deepseek_v4
vocab_size: 16384
context_length: 1024
d_model: 640
num_layers: 8
num_heads: 10
num_kv_heads: 1
d_ff: 512
expert_d_ff: 512
tie_embeddings: true
norm_type: rmsnorm
activation: swiglu
ffn_type: moe
attention_type: hybrid
position_encoding: rope
rope_theta: 1000000.0
partial_rotary_factor: 0.25
qk_norm: true
local_window: 128
num_experts: 8
active_experts: 2
shared_experts: 1
router_score: sqrtsoftplus
residual_streams: 2
compressed_topk: 64
compression_ratio: 4
mtp_layers: 1
mtp_loss_weight: 0.1
```

`scripts/inspect_model.py --device cpu` 的当前参数统计：

```text
total_parameters: 105554600
embedding_parameters: 10485760
non_embedding_parameters: 95068840
active_parameters_per_token: 58368680
```

`active_parameters_per_token` 明显小于 total parameters，因为每个 token 只激活
部分 routed experts。

## Attention schedule

实现文件：

- `src/llm_lab/models/deepseek_v4/model.py`
- `src/llm_lab/models/deepseek_v4/attention.py`

如果配置里没有显式给出 `layer_types`，默认按以下 pattern 循环：

```text
sliding_attention
compressed_sparse_attention
sliding_attention
heavily_compressed_attention
```

8 层正式配置因此对应：

```text
0: sliding_attention
1: compressed_sparse_attention
2: sliding_attention
3: heavily_compressed_attention
4: sliding_attention
5: compressed_sparse_attention
6: sliding_attention
7: heavily_compressed_attention
```

### sliding_attention

`sliding_attention` 使用共享 `MultiHeadAttention`：

- query heads: `10`
- KV heads: `1`
- head dim: `64`
- local window: `128`
- RoPE: partial RoPE
- QK-Norm: enabled

这保留了 DeepSeek V4 中 small-KV-head local attention 的方向。

### compressed_sparse_attention

`compressed_sparse_attention` 使用 `CompressedHybridAttention`：

- Q 使用 full query heads。
- K/V 使用 single shared KV head。
- K/V 按 `compression_ratio = 4` 做均值池化压缩。
- 使用 causal compressed positions。
- 使用 top-k selection，正式配置为 `compressed_topk = 64`。
- 包含 learnable attention sink 参数。

这是 CSA-like 的纯 PyTorch 缩放实现。它保留 compressor/indexer/top-k selection
语义，但没有复现官方 Lightning Indexer、DSA Indexer 或高性能 sparse kernel。

### heavily_compressed_attention

`heavily_compressed_attention` 同样使用 `CompressedHybridAttention`，但：

- compression ratio 加倍。
- 不使用 top-k mask，所有压缩后的 entries 都可参与 attention。

它对应 HCA-like 的长程压缩注意力角色。

## MoE schedule

实现文件：

- `src/llm_lab/models/deepseek_v4/moe.py`
- `src/llm_lab/modules/moe.py`

如果配置里没有显式给出 `moe_layer_types`，默认 schedule 是：

```text
0: hash_moe
1: hash_moe
2-7: moe
```

### Hash-MoE

前两层使用 `HashMoEFeedForward`：

- token id 通过 `token_id % num_experts` 映射到 expert。
- active expert 数为 `2`。
- hash expert 权重均分。
- shared expert 始终参与。

这是 hash-routed bootstrap 的简化实现，用于保留 early hash routing 的结构概念。

### Routed MoE

后续层使用 `MoEFeedForward`：

- router 是线性 gate。
- 对每个 token 选择 top-2 experts。
- router score 使用 `sqrtsoftplus`。
- shared expert 始终参与。
- 每个 expert 是 SwiGLU FFN。

当前 active parameter accounting 会把 router、top-k active experts 和 shared
experts 计入每 token 激活参数。

## Multi-stream residual

DeepSeek V4 公开资料中的 mHC 在当前项目里用 `MultiStreamResidual` 做简化复现。
当 `residual_streams > 1` 时，残差更新为：

```text
x = x + update + Linear(update) / residual_streams
```

它不是完整 mHC 数学形式，但保留了“更新不仅直接加回主 stream，还经过额外
stream mixing”的接口和实验开关。正式配置设置 `residual_streams = 2`。

## MTP 状态

配置和模型里已经有 `mtp_layers = 1`，实现中存在 `mtp_heads`。当前 `forward()`
仍只返回主 next-token logits，训练 loss 当前也是普通 cross entropy。

因此当前状态是：MTP/next-n prediction 接口已经保留，但 auxiliary loss 尚未接入
训练循环。

## 复现边界

保留的真实拓扑：

- MoE FFN path。
- top-k routed experts。
- shared experts。
- hash-routed early MoE。
- `sqrtsoftplus` router scoring。
- single KV head / small KV-head attention。
- partial RoPE。
- sliding / CSA-like / HCA-like attention schedule。
- attention sink。
- simplified multi-stream residual。
- MTP head 接口。
- active parameters per token accounting。

明确简化的部分：

- 不复现官方 exact layer counts、expert counts、hidden sizes 或训练 recipe。
- CSA/HCA 是纯 PyTorch 压缩注意力，不是官方 sparse/indexer kernel。
- mHC 是简化 multi-stream residual，不是完整 manifold-constrained hyper-connection。
- MTP auxiliary loss 还没有接入训练循环。
- 没有实现 grouped low-rank output projection、YaRN 长上下文扩展或部署 cache。

## 实验价值

`deepseek_v4` 应该和 dense 模型分开看两个口径：total parameters 接近 100M，
但 active parameters per token 只有约 58M。它的实验价值在于观察 MoE 和压缩
attention 是否能在相似总参数预算下提供更低激活计算、更高吞吐或不同的 loss
曲线。对它的报告必须同时列出 total parameters、active parameters per token、
tokens/sec、显存和 loss。
