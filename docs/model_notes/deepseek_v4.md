# deepseek_v4 架构记录

本项目的 `deepseek_v4` 是 DeepSeek V4 公开语言模型拓扑的 100M 规模缩放复现，
不复现官方权重、训练数据或工业级 kernel。

## 公开依据

主要依据：

- Hugging Face Transformers docs: https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4
- NVIDIA Megatron Bridge docs: https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/deepseek/deepseek-v4.html

截至 2026-05-31 可公开核对的信息：

- DeepSeek V4 是 MoE language model。
- 架构从 DeepSeek V3 的 MLA 转向 hybrid local + long-range attention。
- decoder block 由 `config.layer_types[i]` 选择三种 attention type：
  - `"sliding_attention"`
  - `"compressed_sparse_attention"` (CSA)
  - `"heavily_compressed_attention"` (HCA)
- CSA 使用低压缩池和 Lightning Indexer，对 compressed entries 做 top-k 选择。
- HCA 使用高压缩池，无 indexer，每个 pooled entry 可进入 attention。
- 三种 attention 共享 backbone：
  - shared K=V Multi-Query Attention，`num_key_value_heads = 1`
  - partial RoPE
  - learnable attention sink
  - grouped low-rank output projection
  - shared sliding-window K=V branch
- V4 使用 Manifold-Constrained Hyper-Connections (mHC)，维护多个 residual streams。
- MoE schedule 由 `config.mlp_layer_types` 控制：
  - `"hash_moe"`：前几层使用 token-id 到 expert-id 的静态映射
  - `"moe"`：后续层使用 top-k routed MoE
- router scoring 使用 `sqrtsoftplus`。
- 默认配置字段包含：
  - `num_key_value_heads = 1`
  - `num_experts_per_tok = 6`
  - `n_routed_experts = 256`
  - `n_shared_experts = 1`
  - `sliding_window = 128`
  - `index_topk = 512`
  - `num_nextn_predict_layers = 1`
  - `partial_rotary_factor`
- NVIDIA 文档还强调 CSA with DSA Indexer、mHC、Hash-Routed MoE、MTP
  with separate `e_proj` / `h_proj` projections、YaRN RoPE 和 grouped output projection。

## 缩放复现要求

`deepseek_v4` 必须保留真实拓扑的关键结构：

- MoE FFN path，包含 routed experts 和 active expert accounting。
- 支持 shared expert 概念。
- 支持 MTP 或兼容接口。
- partial RoPE。
- 单 KV head 或极少 KV heads。
- layer-level attention schedule，至少覆盖 sliding / CSA-like / HCA-like 三种 layer type。
- hash-routed MoE bootstrap 概念。
- mHC 或一个明确标注为简化版的 multi-stream residual mixer。

允许缩放：

- hidden size、层数、expert 数、active expert 数、compress rates、context length。
- CSA/HCA 第一版可以用纯 PyTorch 简化实现，但接口必须保留 compressor/indexer 语义。
- mHC 第一版可以采用简化 multi-stream mixing，但不能退化成普通 residual 而不记录原因。

不允许：

- 把 `deepseek_v4` 简化成普通 dense Transformer。
- 把 `deepseek_v4` 简化成普通 MoE Transformer 后忽略 CSA/HCA、partial RoPE、MTP、mHC 等公开结构。
- 只报告 total parameters 而不报告 active parameters per token。
