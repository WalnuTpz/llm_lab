# LLM Lab

LLM Lab 是一个面向语言模型架构复现与对比实验的研究型代码库。项目以约
100M 参数规模为主要实验尺度，在统一的 Python 包、配置系统、数据/tokenizer
流程、训练/生成/benchmark 脚本和 CPU smoke tests 下，对比四类代表性语言模型
架构：

- `original_transformer`
- `modern_decoder`
- `qwen36`
- `deepseek_v4`

项目关注架构层面的缩放复现：保留核心拓扑、模块接口和可比较的实验口径。项目
不以复现官方权重、官方训练数据、post-training 流程或工业级 CUDA kernel 为目标。

## 项目范围

当前主实验规模约为 100M 参数。正式训练预期在云端 GPU 环境执行；本地开发环境
只要求能够通过 CPU smoke tests。

相关文档：

- [docs/implementation_requirements.md](docs/implementation_requirements.md)：实现要求和模型定位。
- [docs/experiment_protocol.md](docs/experiment_protocol.md)：四模型公平比较协议。
- [docs/data_tokenizer_plan.md](docs/data_tokenizer_plan.md)：OpenWebText 和 tokenizer 路线。
- [docs/cloud_training.md](docs/cloud_training.md)：云端训练 runbook。

## 架构

`original_transformer` 是历史基线模型，采用 decoder-only causal LM 接口，并使用
full MHA、sinusoidal position、LayerNorm 和 ReLU FFN。

`modern_decoder` 是现代 dense decoder 基线，采用 RMSNorm、RoPE、SwiGLU、GQA
和可选 QK-Norm。

`qwen36` 是 Qwen3.6 公开语言模型拓扑的缩放复现，保留 3:1 的
linear-mixer/full-attention layer schedule，并实现 Gated DeltaNet-like mixer 与
Gated Attention-like full attention。

`deepseek_v4` 是 DeepSeek V4 公开语言模型拓扑的缩放复现，保留 MoE、shared
experts、MTP heads、partial RoPE、small KV-head attention，以及 CSA/HCA-like
compressed attention 接口。

架构说明：

- [docs/model_notes/original_transformer.md](docs/model_notes/original_transformer.md)
- [docs/model_notes/modern_decoder.md](docs/model_notes/modern_decoder.md)
- [docs/model_notes/qwen36.md](docs/model_notes/qwen36.md)
- [docs/model_notes/deepseek_v4.md](docs/model_notes/deepseek_v4.md)

## 目录结构

```text
configs/                    # 100M-scale 正式实验配置
docs/                       # 实现要求、架构说明、实验协议、云端流程
scripts/                    # CLI 入口
src/llm_lab/                # Python 包
tests/                      # CPU smoke tests 和小配置 fixture
```

以下运行产物默认不进入 Git：

```text
data/
artifacts/
checkpoints/
runs/
wandb/
```

## 环境

项目使用 `uv` 管理环境：

```bash
uv sync
```

如果在 Windows UNC 路径下使用 `uv` 遇到问题，建议从 WSL 内运行命令，或使用
WSL workspace 中已有的 `.venv`。

RTX PRO 6000 Blackwell 等 `sm_120` GPU 需要支持 Blackwell 的 PyTorch wheel。
当前依赖已升级到 `torch~=2.8.0`，Linux CUDA 依赖使用 CUDA 12.8 线。

## 测试

```bash
uv run pytest
```

测试集是 CPU-only，使用 `tests/fixtures/configs/` 中的小模型配置，覆盖以下内容：

- config loading
- model construction
- forward / loss / backward
- tokenizer train/load/tokenize
- parameter accounting
- checkpoint / resume
- metrics summary

## 查看模型参数

查看单个配置的参数统计：

```bash
uv run python scripts/inspect_model.py --config configs/modern_decoder.yaml --device cpu
```

汇总四个正式配置的参数统计：

```bash
uv run python scripts/inspect_all_models.py
```

`inspect_all_models.py` 默认使用 `meta` device，不实际分配 100M 权重。

## Smoke Train

使用随机 token 进行 smoke train，不需要真实数据集：

```bash
uv run python scripts/train.py \
  --config tests/fixtures/configs/modern_decoder_small.yaml \
  --smoke \
  --device cpu \
  --max-iters 1
```

训练脚本会写入：

```text
runs/<architecture>/metrics.jsonl
checkpoints/<architecture>/latest.pt
```

断点续训：

```bash
uv run python scripts/train.py --config configs/modern_decoder.yaml --resume
```

## Tokenizer 和数据

最小 smoke pipeline 保留 `ByteTokenizer`。

正式实验使用 OpenWebText 训练 16k byte-level BPE tokenizer。默认训练后端为
HuggingFace `tokenizers`；项目内纯 Python BPE 实现作为参考实现和 fallback 保留。

```bash
uv run python scripts/train_tokenizer.py \
  --input data/raw/openwebtext/train.jsonl \
  --output-dir data/tokenizers/bpe_16k \
  --vocab-size 16384 \
  --backend tokenizers \
  --jsonl-field text \
  --max-chars 200000000
```

使用项目内 Python fallback：

```bash
uv run python scripts/train_tokenizer.py \
  --input data/raw/openwebtext/train.jsonl \
  --output-dir data/tokenizers/bpe_16k_python \
  --vocab-size 16384 \
  --backend python
```

tokenize 数据集：

```bash
uv run python scripts/tokenize_dataset.py \
  --input data/raw/openwebtext/train.jsonl \
  --tokenizer-path data/tokenizers/bpe_16k \
  --output data/tokens/bpe_16k/train.npy \
  --jsonl-field text \
  --add-eot \
  --dtype uint16
```

## 云端运行

按模型名启动正式配置训练：

```bash
uv run python scripts/run_experiment.py modern_decoder --device cuda --max-iters 1000
```

可选模型名：

```text
original_transformer
modern_decoder
qwen36
deepseek_v4
```

预览命令而不执行：

```bash
uv run python scripts/run_experiment.py qwen36 --device cuda --max-iters 1000 --dry-run
```

汇总训练结果：

```bash
uv run python scripts/summarize_runs.py
```

## 生成

```bash
uv run python scripts/generate.py \
  --config configs/modern_decoder.yaml \
  --prompt "hello" \
  --max-new-tokens 16 \
  --device cpu
```

在 PowerShell 中传逗号分隔 token ids 时，需要加引号：

```bash
uv run python scripts/generate.py \
  --config tests/fixtures/configs/modern_decoder_small.yaml \
  --prompt-ids "1,2"
```

## Benchmark

```bash
uv run python scripts/benchmark.py \
  --config tests/fixtures/configs/qwen36_small.yaml \
  --device cpu \
  --steps 3
```

GPU benchmark 会额外报告 CUDA peak memory。
