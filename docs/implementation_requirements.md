# LLM 架构实验台实现要求

## 项目目标

本项目定位为 **100M 规模的 LLM 架构复现实验台**。

目标是在同一套 tokenizer、数据管线、训练脚本和评测/benchmark 框架下，
比较四类语言模型架构：

1. `original_transformer`
2. `modern_decoder`
3. `qwen36`
4. `deepseek_v4`

本项目复现的是架构思想、模块拓扑和可比较的工程接口，不复现官方权重、
官方训练数据、后训练流程或工业级推理 kernel。

## 硬性约束

- 只使用一个主实验规模：100M 参数左右。
- 目标硬件：单张 RTX 4090。
- 训练预算：每个模型训练时间不超过 10 小时。
- 四个模型必须尽量使用相同的 tokenizer、数据集、context length、optimizer、
  scheduler、日志和评测流程。
- 不把 `tiny` 或 debug 规模作为正式配置。单元测试内部可以使用小 shape，
  但项目配置文件应该代表 100M 规模实验。

## 复现原则

`qwen36` 和 `deepseek_v4` 要尽量保留真实模型公开架构的形状，只把宽度、深度、
head 数、expert 数等缩小到 100M 预算内。

允许的简化：

- 使用纯 PyTorch 实现，而不是自定义 CUDA kernel。
- 缩小 `d_model`、层数、head 数、expert 数和 FFN 宽度。
- 对 compressed / sparse / linear attention 使用简化 kernel，只要数据流、
  模块接口和公开结构保持接近。
- 如果 4090 训练预算不允许，可以降低训练 context length。

不允许的简化：

- 把 `qwen36` 简化成普通 dense Transformer 后只改名字。
- 把 `deepseek_v4` 简化成普通 MoE Transformer 后只改名字。
- 用不同 tokenizer、数据或优化规则比较四个模型。
- MoE 模型只报告 total parameters，不报告 active parameters per token。

## 四个模型的定位

### original_transformer

定位：历史基线。

实现目标：

- 为了公平比较，使用 decoder-only LM 训练接口。
- 内部组件尽量保持原始 Transformer 风格：
  - full multi-head attention
  - sinusoidal positional encoding
  - LayerNorm
  - ReLU feed-forward network
  - 不使用 RoPE、RMSNorm、SwiGLU、GQA

它回答的问题是：在相同训练规模下，现代 LLM block 设计到底带来多少差异。

### modern_decoder

定位：稳定的现代 dense decoder baseline。

实现目标：

- decoder-only autoregressive LM
- pre-norm RMSNorm
- RoPE
- SwiGLU FFN
- GQA
- optional QK-Norm
- optional KV-cache generation path

这个模型应该作为比较 `qwen36` 和 `deepseek_v4` 的稳定现代基线。

### qwen36

定位：Qwen3.6 公开架构拓扑的 100M 缩放复现。

实现目标：

- 保留 Qwen3.6 风格的 hybrid layer schedule。
- 保留重复层模式：
  - 3 个 linear-mixer layer
  - 1 个 full-attention layer
- 实现 Gated DeltaNet-like linear mixer，不能直接替换成普通 attention。
- 实现 Gated Attention-like full attention block。
- 配置接口尽量接近 Qwen 风格，包括：
  - `full_attention_interval`
  - `layer_types`
  - attention head count
  - KV head count
  - linear mixer settings
  - optional MTP settings

缩放策略：

- 使用 100M 规模的 hidden size、层数、head 数和 FFN 宽度。
- 即使官方维度被缩小，也要保留 3:1 的 mixer schedule。

### deepseek_v4

定位：DeepSeek V4 公开架构拓扑的 100M 缩放复现。

实现目标：

- MoE FFN path with routed experts。
- 如果 100M 预算允许，支持 shared expert。
- 同时报告 total parameters 和 active parameters per token。
- MTP module 或兼容接口。
- partial RoPE。
- 根据预算使用单 KV head 或很少的 KV heads。
- 设计 hybrid/compressed attention 接口，后续可以演进到 CSA/HCA-like 行为。
- 配置字段尽量接近 DeepSeek 风格，覆盖 sparse/compressed attention、routing、
  experts、active experts、MTP 等设置。

缩放策略：

- 100M 预算下优先保证 active compute 可比较。
- 如果使用 MoE，total parameters 可以略高于 dense baseline，但必须保持单卡 4090
  可训练。

## 参数和性能统计要求

每次模型运行都应该记录：

- total parameters
- trainable parameters
- active parameters per token
- embedding parameters
- non-embedding parameters
- peak GPU memory
- tokens/sec
- loss curve
- validation perplexity
- training/evaluation context length

MoE 模型比较时必须区分：

- capacity-matched：total parameters 接近
- compute-matched：active parameters per token 接近

本项目第一阶段优先做 compute-matched comparison。

## 配置系统要求

顶层正式配置文件：

- `configs/original_transformer.yaml`
- `configs/modern_decoder.yaml`
- `configs/qwen36.yaml`
- `configs/deepseek_v4.yaml`

所有脚本都应该通过以下路径构建模型：

```text
config -> registry -> model
```

训练、生成和 benchmark 代码不应该直接关心模型内部细节，只依赖稳定模块接口。

## 目录要求

最终源码使用 `src/llm_lab/` 作为 Python package。

核心目录：

- `src/llm_lab/config/`
- `src/llm_lab/models/`
- `src/llm_lab/models/original_transformer/`
- `src/llm_lab/models/modern_decoder/`
- `src/llm_lab/models/qwen36/`
- `src/llm_lab/models/deepseek_v4/`
- `src/llm_lab/modules/`
- `src/llm_lab/data/`
- `src/llm_lab/training/`
- `src/llm_lab/generation/`
- `src/llm_lab/evaluation/`
- `src/llm_lab/benchmarking/`
- `src/llm_lab/utils/`

顶层辅助目录：

- `configs/`
- `docs/`
- `docs/model_notes/`
- `scripts/`
- `tests/`

运行时输出目录应该继续被 Git 忽略：

- `data/`
- `artifacts/`
- `checkpoints/`
- `runs/`
- `wandb/`

## 开发顺序

1. 创建 config schema 和 model registry。
2. 把当前 CS336 风格实现迁移成 `modern_decoder`。
3. 实现 `original_transformer` 基线。
4. 增加共享模块：attention、RoPE、norms、FFN、KV-cache、参数统计。
5. 实现 `qwen36` 的 3:1 linear-mixer/full-attention schedule。
6. 分阶段实现 `deepseek_v4`：
   - MoE routing 和 expert accounting
   - MTP
   - partial RoPE 和 small KV-head attention
   - compressed/hybrid attention interface
7. 增加统一 benchmark：速度、显存、参数量、loss 曲线。

## 非目标

- 训练 frontier-quality 模型。
- 匹配 Qwen 或 DeepSeek 官方权重。
- 复现专有训练数据或后训练 pipeline。
- 第一版就实现生产级 CUDA kernel。
- 在 100M 规模比较稳定前支持大量模型尺寸。
