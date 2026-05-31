# modern_decoder 架构说明

`modern_decoder` 是本项目的稳定现代 dense decoder 基线。它不绑定某个具体模型
家族，而是组合当前小到中等规模 LLM 中最常见的一组 decoder-only 组件，用来和
`original_transformer`、`qwen36`、`deepseek_v4` 做公平对比。

## 定位

这个模型回答的问题是：如果不引入 linear attention、hybrid attention 或 MoE，
只采用成熟的现代 dense decoder 设计，在约 100M 参数规模下可以达到怎样的训练
稳定性和速度。

它是后续实验最适合作为第一条训练基线的模型，因为结构清楚、实现风险最低、
active parameters 和 total parameters 一致。

## 当前正式配置

配置文件：`configs/modern_decoder.yaml`

```yaml
architecture: modern_decoder
vocab_size: 16384
context_length: 1024
d_model: 768
num_layers: 14
num_heads: 12
num_kv_heads: 4
d_ff: 2048
tie_embeddings: true
norm_type: rmsnorm
activation: swiglu
ffn_type: swiglu
attention_type: gqa
position_encoding: rope
rope_theta: 1000000.0
partial_rotary_factor: 1.0
qk_norm: true
```

`scripts/inspect_model.py --device cpu` 的当前参数统计：

```text
total_parameters: 100685568
embedding_parameters: 12582912
non_embedding_parameters: 88102656
active_parameters_per_token: 100685568
```

这是 dense 模型，所以每个 token 的 active parameters 等于 total parameters。

## Block 结构

实现文件：`src/llm_lab/models/modern_decoder/model.py`

每层 block 的结构是标准 pre-norm decoder：

```text
x = x + GQA(RMSNorm(x), RoPE, QK-Norm)
x = x + SwiGLU(RMSNorm(x))
```

attention 使用共享模块 `llm_lab.modules.MultiHeadAttention`：

- query heads: `num_heads = 12`
- KV heads: `num_kv_heads = 4`
- head dim: `d_model / num_heads = 64`
- position: RoPE
- `partial_rotary_factor = 1.0`，当前正式配置对完整 head dim 使用 RoPE。
- `qk_norm = true`，对 Q/K 做 RMS unit normalization。
- 默认 full causal attention，不启用 local window。

FFN 使用 `SwiGLUFeedForward`：

```text
SwiGLU(x) = W2(silu(W1(x)) * W3(x))
```

## 复现边界

保留的现代 decoder 结构：

- decoder-only causal LM。
- pre-norm RMSNorm。
- RoPE。
- GQA。
- SwiGLU FFN。
- tied token embedding / LM head。
- 可选 QK-Norm。

没有刻意复现的部分：

- 某个具体模型家族的精确层数、head 数或 FFN ratio。
- tokenizer 训练配方。
- FlashAttention、paged KV cache、fused RMSNorm、fused SwiGLU 等工业 kernel。
- sliding window / long context trick。

## 实验价值

`modern_decoder` 应该先被训练和调参，因为它是后续所有复杂架构的参照物。如果
它在相同数据和训练脚本下不能稳定下降，优先排查数据、tokenizer、optimizer 和
训练循环，而不是先怀疑 `qwen36` 或 `deepseek_v4` 的复杂模块。
