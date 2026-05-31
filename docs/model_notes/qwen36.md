# qwen36 架构记录

本项目的 `qwen36` 是 Qwen3.6 语言主干的 100M 规模缩放复现，不复现官方权重、
视觉 encoder、训练数据或部署 kernel。

## 公开依据

主要依据：

- Hugging Face model card: https://huggingface.co/Qwen/Qwen3.6-27B
- Hugging Face config: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json

截至 2026-05-31 可公开核对的信息：

- 模型卡称 Qwen3.6-27B 是 causal language model with vision encoder。
- 语言模型参数量为 27B，hidden dimension 为 5120。
- 语言模型有 64 层。
- hidden layout 为 `16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))`。
- Gated DeltaNet 使用 linear attention heads：V 为 48 heads，QK 为 16 heads，head dim 为 128。
- Gated Attention 使用 Q heads 24、KV heads 4、head dim 256。
- FFN intermediate dimension 为 17408。
- MTP 使用 multi-step training。
- native context length 为 262144，并可通过 YaRN 扩展到约 1010000。
- `config.json` 中 `text_config.full_attention_interval = 4`。
- `config.json` 中 `text_config.layer_types` 按 3 个 `"linear_attention"` 加 1 个
  `"full_attention"` 重复。
- `config.json` 中 `mtp_num_hidden_layers = 1`。
- `config.json` 中 `partial_rotary_factor = 0.25`，`rope_theta = 10000000`。

## 缩放复现要求

`qwen36` 必须保留真实拓扑的关键结构：

- 3:1 的 `linear_attention` / `full_attention` layer schedule。
- linear layer 使用 Gated DeltaNet-like mixer，而不是普通 self-attention。
- full layer 使用 Gated Attention-like full attention。
- 每个 token mixer 后接 FFN。
- 配置字段保留 `full_attention_interval`、`layer_types`、linear mixer heads、
  full attention heads、KV heads、MTP 等概念。

允许缩放：

- hidden size、层数、heads、FFN 宽度、context length。
- Gated DeltaNet 的 kernel 可以用纯 PyTorch 简化实现。
- 视觉 encoder 第一版可以不实现，默认 text-only。

不允许：

- 把 75% 的 linear_attention 层替换成普通 full attention。
- 去掉 3:1 layer schedule。
- 只实现一个普通 Transformer 后命名为 qwen36。
