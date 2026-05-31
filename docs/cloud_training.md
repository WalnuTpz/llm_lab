# Cloud Training Guide

本文档描述在云端 GPU 机器上运行 LLM Lab 四模型实验的推荐流程。本地开发环境不
需要下载 OpenWebText，也不需要训练 100M 模型。

## 1. Clone And Install

```bash
git clone git@github.com:WalnuTpz/llm_lab.git
cd llm_lab
uv sync
```

确认代码和配置：

```bash
uv run python scripts/inspect_all_models.py
uv run pytest
```

`inspect_all_models.py` 默认使用 `meta` device，不会分配 100M 权重。

## 2. Prepare OpenWebText

把 OpenWebText 准备成本地 JSONL，推荐路径：

```text
data/raw/openwebtext/train.jsonl
data/raw/openwebtext/val.jsonl
```

每行是一条 JSON object，默认字段名：

```json
{"text": "..."}
```

如果字段名不是 `text`，后续命令用 `--jsonl-field` 指定。

## 3. Train 16k Byte-Level BPE Tokenizer

只需要从 train split 中抽样训练 tokenizer。先从 100M 到 300M chars 开始，不必
一次吃完整 OpenWebText。

```bash
uv run python scripts/train_tokenizer.py \
  --input data/raw/openwebtext/train.jsonl \
  --output-dir data/tokenizers/bpe_16k \
  --vocab-size 16384 \
  --name openwebtext_bpe_16k \
  --backend tokenizers \
  --jsonl-field text \
  --max-chars 200000000
```

`tokenizers` 是默认 backend，使用 HuggingFace Rust 实现。项目内纯 Python BPE
实现仍然保留，可用 `--backend python` 做小样本调试或学习用途，但不建议作为
OpenWebText 正式训练默认路径。

生成文件：

```text
data/tokenizers/bpe_16k/tokenizer.json
data/tokenizers/bpe_16k/tokenizer_config.yaml
```

## 4. Tokenize Train And Val

```bash
uv run python scripts/tokenize_dataset.py \
  --input data/raw/openwebtext/train.jsonl \
  --tokenizer-path data/tokenizers/bpe_16k \
  --output data/tokens/bpe_16k/train.npy \
  --jsonl-field text \
  --add-eot \
  --dtype uint16

uv run python scripts/tokenize_dataset.py \
  --input data/raw/openwebtext/val.jsonl \
  --tokenizer-path data/tokenizers/bpe_16k \
  --output data/tokens/bpe_16k/val.npy \
  --jsonl-field text \
  --add-eot \
  --dtype uint16
```

正式 configs 已经指向这些路径：

```yaml
data:
  train_data: data/tokens/bpe_16k/train.npy
  val_data: data/tokens/bpe_16k/val.npy
  tokenizer_path: data/tokenizers/bpe_16k
  dataset_dtype: uint16
```

## 5. Choose A Token Budget

训练比较按 token budget，不按 epoch。`max_iters` 是 optimizer step 数。实际
训练 tokens 计算方式：

```text
tokens = max_iters * batch_size * context_length * grad_accum_steps
```

四个模型要使用相同 token budget。先用 `modern_decoder` 估算 4090 上的吞吐，再
确定 main run 的 `max_iters`。

建议阶段：

```text
debug: 1M tokens
pilot: 50M tokens
main: 300M or 500M tokens
```

## 6. Run One Experiment

按模型名启动正式 config：

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

断点续训：

```bash
uv run python scripts/run_experiment.py modern_decoder --device cuda --max-iters 2000 --resume
```

传入 `--resume` 且不带路径时，会读取该模型 `checkpoint_dir/latest.pt`。

## 7. Run All Four Models

当前项目没有自动调度队列；建议在云端按相同 token budget 顺序运行：

```bash
uv run python scripts/run_experiment.py original_transformer --device cuda --max-iters <N>
uv run python scripts/run_experiment.py modern_decoder --device cuda --max-iters <N>
uv run python scripts/run_experiment.py qwen36 --device cuda --max-iters <N>
uv run python scripts/run_experiment.py deepseek_v4 --device cuda --max-iters <N>
```

如果要预览命令：

```bash
uv run python scripts/run_experiment.py qwen36 --device cuda --max-iters <N> --dry-run
```

## 8. Summarize Runs

训练脚本会写：

```text
runs/<architecture>/metrics.jsonl
checkpoints/<architecture>/latest.pt
```

汇总：

```bash
uv run python scripts/summarize_runs.py
```

JSONL 输出：

```bash
uv run python scripts/summarize_runs.py --format jsonl > artifacts/run_summary.jsonl
```

汇总结果会包含：

- architecture
- last step
- tokens
- last train loss
- last val loss
- tokens/sec
- total parameters
- active parameters per token
- checkpoint count

## 9. Minimum Cloud Checklist

正式跑四模型前，至少确认：

- `uv run pytest` 通过。
- `scripts/inspect_all_models.py` 参数表符合预期。
- `data/tokenizers/bpe_16k/tokenizer.json` 存在。
- `data/tokens/bpe_16k/train.npy` 和 `val.npy` 存在。
- 四个 config 的 `batch_size`、`context_length`、`grad_accum_steps` 能形成相同
  token budget。
- `modern_decoder` 的 debug run 能正常写 `metrics.jsonl` 和 `latest.pt`。
