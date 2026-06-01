# qwen36 架构说明

`qwen36` 是本项目对 Qwen3.6 公开语言模型拓扑的 100M-scale 缩放复现。它的
目标不是复现官方权重、训练数据、视觉 encoder 或生产 kernel，而是在本项目的
统一接口下尽量保留 Qwen3.6 text backbone 的关键结构。

## 公开架构依据

主要依据：

- Hugging Face model card: https://huggingface.co/Qwen/Qwen3.6-27B
- Hugging Face config: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json

截至 2026-05-31，可公开核对的关键点包括：

- text model 是 causal LM。
- hidden layout 是 3 个 Gated DeltaNet layer 后接 1 个 Gated Attention layer，
  重复形成 3:1 的 layer schedule。
- `full_attention_interval = 4`。
- `layer_types` 按 `"linear_attention"`、`"linear_attention"`、
  `"linear_attention"`、`"full_attention"` 重复。
- Gated DeltaNet 承担大多数 layer 的 linear mixer 角色。
- Gated Attention 承担周期性 full attention 角色。
- `mtp_num_hidden_layers = 1`。
- `partial_rotary_factor = 0.25`。
- `rope_theta = 10000000`。

## 定位

这个模型回答的问题是：在约 100M 参数规模下，把大部分 full attention 层替换成
linear mixer，并周期性保留 full attention，会如何影响训练速度、显存、loss 和
长上下文行为。

它不是“普通 Transformer 改名”。当前实现保留了 3:1 的 linear/full schedule，
并区分 Gated DeltaNet-like mixer 和 Gated Attention-like full attention。

## 当前正式配置

配置文件：`configs/qwen36.yaml`

```yaml
architecture: qwen36
vocab_size: 16384
context_length: 1024
d_model: 720
num_layers: 12
num_heads: 12
num_kv_heads: 4
d_ff: 1920
tie_embeddings: true
norm_type: rmsnorm
activation: swiglu
ffn_type: swiglu
attention_type: hybrid
position_encoding: rope
rope_theta: 10000000.0
partial_rotary_factor: 0.25
qk_norm: true
full_attention_interval: 4
mtp_layers: 1
mtp_loss_weight: 0.1
```

`scripts/inspect_model.py --device cpu` 的当前参数统计：

```text
total_parameters: 102407760
embedding_parameters: 11796480
non_embedding_parameters: 90611280
active_parameters_per_token: 102407760
```

当前实现不是 MoE，所以每个 token 的 active parameters 等于 total parameters。

## Layer schedule

实现文件：

- `src/llm_lab/models/qwen36/model.py`
- `src/llm_lab/models/qwen36/mixers.py`

默认 `qwen36_layer_types()` 根据 `full_attention_interval = 4` 生成 12 层：

```text
0:  linear_attention
1:  linear_attention
2:  linear_attention
3:  full_attention
4:  linear_attention
5:  linear_attention
6:  linear_attention
7:  full_attention
8:  linear_attention
9:  linear_attention
10: linear_attention
11: full_attention
```

每个 block 的公共结构是：

```text
x = x + Mixer(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

## Gated DeltaNet-like mixer

`linear_attention` 层使用 `GatedDeltaNetMixer`。当前实现是纯 PyTorch 版本，核心
流程是：

```text
q = elu(Wq(x)) + 1
k = elu(Wk(x)) + 1
v = Wv(x)
gate = sigmoid(Wg(x))

kv_state = cumsum(k outer v)
k_state = cumsum(k)
y_t = q_t * kv_state_t / (q_t * k_state_t)
y = Wo(y * gate)
```

它保留了 causal linear mixer、门控和按时间累积状态的结构思想，但没有复现官方
Gated DeltaNet 的高性能 kernel、精确更新规则和所有 head layout 细节。

当前缩放版本使用 `d_model=720`、`num_heads=12`，所以 linear mixer head dim 为
60。官方 Qwen3.6 中 DeltaNet 的 QK/V head layout 与本项目缩放配置不同，这是
有意缩放。

## Gated Attention-like full attention

`full_attention` 层使用 `GatedFullAttention`：

```text
y = GQA(x, RoPE, QK-Norm) * sigmoid(Wg(x))
```

其中 GQA 配置为：

- query heads: `12`
- KV heads: `4`
- head dim: `60`
- RoPE: `partial_rotary_factor = 0.25`
- RoPE theta: `10000000`
- QK-Norm: enabled

full attention 只出现在每 4 层的最后一层，用来保留周期性的全局信息混合。

## MTP 状态

配置和模型里已经有 `mtp_layers = 1`，实现中存在 `mtp_heads`。普通 `forward()`
仍只返回主 next-token logits，以保持 generation/benchmark 接口简单。

训练脚本会在 `mtp_loss_weight > 0` 时调用 `forward_with_mtp()`，额外预测
`token_{t+2}`，并使用：

```text
loss = next_token_loss + mtp_loss_weight * mtp_loss
```

当前实现是轻量 auxiliary MTP head，不是官方完整 MTP module。

## 复现边界

保留的真实拓扑：

- 3:1 linear attention / full attention schedule。
- Gated DeltaNet-like linear mixer。
- Gated full attention。
- RMSNorm + SwiGLU decoder block。
- partial RoPE。
- QK-Norm。
- MTP head 接口。

明确简化的部分：

- 不实现视觉 encoder。
- 不复现官方 27B hidden size、64 层、head layout 和长上下文配置。
- 不复现 YaRN 扩展、训练数据、post-training 或部署 kernel。
- DeltaNet 是教学/实验用纯 PyTorch mixer，不是官方 kernel。
- MTP 是轻量 auxiliary head，不复现官方完整 MTP block。

## 实验价值

`qwen36` 应该主要和 `modern_decoder` 对比。它们都是 dense 模型，参数量接近，
区别在于 `qwen36` 用 75% linear mixer + 25% full attention 的 hybrid backbone
替代每层 full attention。关键观测指标应包括 tokens/sec、显存、loss 曲线以及
上下文长度增长时的行为。
