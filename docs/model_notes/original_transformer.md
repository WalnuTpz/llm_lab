# original_transformer 架构说明

`original_transformer` 是本项目的历史基线模型。它不是复现某一个具体开源
checkpoint，而是把原始 Transformer decoder 的核心组件放到统一的 causal LM
接口里，用来和后面的现代 decoder、hybrid/linear 模型、MoE 模型做可控对比。

## 定位

这个模型回答的问题是：如果只使用较早期 Transformer 常见组件，在相同
tokenizer、训练脚本、数据集和大约 100M 参数规模下，表现和效率会是什么水平。

因此它刻意保持简单：

- full multi-head self-attention，不使用 GQA/MQA。
- sinusoidal position embedding，不使用 RoPE。
- LayerNorm，不使用 RMSNorm。
- ReLU FFN，不使用 SwiGLU。
- dense FFN，不使用 MoE。
- decoder-only causal LM，不实现 encoder-decoder cross-attention。

decoder-only 是为了让四个模型共享同一套训练、生成和 benchmark 接口。它不是
完整的原始 encoder-decoder Transformer。

## 当前正式配置

配置文件：`configs/original_transformer.yaml`

```yaml
architecture: original_transformer
vocab_size: 16384
context_length: 1024
d_model: 768
num_layers: 12
num_heads: 12
num_kv_heads: 12
d_ff: 3072
tie_embeddings: true
norm_type: layernorm
activation: relu
ffn_type: relu
attention_type: mha
position_encoding: sinusoidal
qk_norm: false
```

`scripts/inspect_model.py --device cpu` 的当前参数统计：

```text
total_parameters: 97555968
embedding_parameters: 12582912
non_embedding_parameters: 84973056
active_parameters_per_token: 97555968
```

这是 dense 模型，所以每个 token 的 active parameters 等于 total parameters。

## Block 结构

实现文件：`src/llm_lab/models/original_transformer/model.py`

每层 block 的结构是：

```text
x = LayerNorm(x + CausalMHA(x))
x = LayerNorm(x + ReLUFFN(x))
```

这是一种 post-norm residual 形式。attention 使用共享模块
`llm_lab.modules.MultiHeadAttention`，其中：

- `num_kv_heads = num_heads`，所以是标准 MHA。
- 使用 causal mask。
- 不使用 RoPE。
- 不使用 QK-Norm。
- 不使用 local window。

position 使用 `SinusoidalPositionEmbedding`，和 token embedding 相加后送入
decoder blocks。

## 复现边界

保留的结构思想：

- 自回归 causal attention。
- full attention 的二次复杂度行为。
- sinusoidal absolute position。
- ReLU 两层 FFN。
- LayerNorm residual block。

项目中没有复现的部分：

- encoder-decoder 架构。
- cross-attention。
- label smoothing、原论文学习率 schedule 等训练细节。
- fused attention kernel 或高性能推理 cache。

## 实验价值

这个模型主要作为低现代化程度的参照组。和 `modern_decoder` 的对比可以隔离出
RoPE、RMSNorm、SwiGLU、GQA、QK-Norm 等现代 decoder 组件的影响；和
`qwen36` / `deepseek_v4` 的对比可以观察 full dense attention 在相同预算下的
loss、tokens/sec 和显存走势。
