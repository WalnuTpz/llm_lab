# Data and Tokenizer Plan

本项目正式实验统一使用 OpenWebText 路线和 16k byte-level BPE tokenizer。当前本地
环境不需要下载或加载 OpenWebText；本地只验证工具链在 tiny fixture 上可运行。
真实 tokenizer 训练、数据 tokenize 和模型训练放到云端机器执行。

## Tokenizer

正式 tokenizer：

```text
type: byte-level BPE
default training backend: HuggingFace tokenizers
vocab_size: 16384
special_tokens:
  <|endoftext|>: 16383
```

选择这个规格的原因：

- byte-level BPE 没有 OOV，能稳定处理英文、中文、符号和代码。
- 16k vocab 与四个 100M-scale 模型配置一致，不会显著改变 embedding 参数量。
- 四个模型必须共享同一个 tokenizer，避免 tokenizer 差异污染架构对比。

项目中的 `ByteTokenizer` 继续保留，只用于 smoke tests 和极小调试。正式实验默认
使用 HuggingFace `tokenizers` 的 Rust backend 训练 byte-level BPE；项目内的纯
Python BPE 实现继续保留为参考实现和 fallback，可通过 `--backend python` 使用。

## Dataset

正式数据集方向是 OpenWebText。推荐云端目录约定：

```text
data/
  raw/
    openwebtext/
      train.jsonl
      val.jsonl
  tokenizers/
    bpe_16k/
      tokenizer.json
      tokenizer_config.yaml
  tokens/
    bpe_16k/
      train.npy
      val.npy
```

JSONL 默认字段名是 `text`。如果云端数据字段不同，用 CLI 的 `--jsonl-field` 指定。

## Cloud Commands

训练 tokenizer 时不需要使用全部 OpenWebText。建议从 train split 抽样固定数量的
字符，例如 100M 到 300M chars，先保证可复现：

```bash
python scripts/train_tokenizer.py \
  --input data/raw/openwebtext/train.jsonl \
  --output-dir data/tokenizers/bpe_16k \
  --vocab-size 16384 \
  --name openwebtext_bpe_16k \
  --backend tokenizers \
  --jsonl-field text \
  --max-chars 200000000
```

如果需要使用项目内纯 Python 参考实现：

```bash
python scripts/train_tokenizer.py \
  --input data/raw/openwebtext/train.jsonl \
  --output-dir data/tokenizers/bpe_16k_python \
  --vocab-size 16384 \
  --backend python \
  --jsonl-field text \
  --max-chars 50000000
```

纯 Python backend 主要用于学习、调试和小规模对照；OpenWebText 正式 tokenizer
训练默认使用 `tokenizers` backend。

tokenize train/val：

```bash
python scripts/tokenize_dataset.py \
  --input data/raw/openwebtext/train.jsonl \
  --tokenizer-path data/tokenizers/bpe_16k \
  --output data/tokens/bpe_16k/train.npy \
  --jsonl-field text \
  --add-eot \
  --dtype uint16

python scripts/tokenize_dataset.py \
  --input data/raw/openwebtext/val.jsonl \
  --tokenizer-path data/tokenizers/bpe_16k \
  --output data/tokens/bpe_16k/val.npy \
  --jsonl-field text \
  --add-eot \
  --dtype uint16
```

`tokenize_dataset.py` 会先统计 token 数，再用 `numpy.open_memmap` 写 `.npy`，避免
把所有 token 同时放进 Python list。

## Config Wiring

正式训练配置应指向同一组 token files：

```yaml
data:
  train_data: data/tokens/bpe_16k/train.npy
  val_data: data/tokens/bpe_16k/val.npy
  tokenizer_path: data/tokenizers/bpe_16k
  dataset_dtype: uint16
```

四个模型的 `model.vocab_size` 必须大于等于 tokenizer vocab size。当前正式配置都
是 `16384`，正好匹配。

## Experiment Budget

架构对比按 token budget，不按 epoch：

```text
debug: 1M tokens
pilot: 50M tokens
main: 300M or 500M tokens
```

正式 token budget 要等 `modern_decoder` 在云端 4090 上跑出实际 tokens/sec 后再
确定。报告中至少列出：

- total parameters
- active parameters per token
- train tokens
- batch size、context length、gradient accumulation
- tokens/sec
- peak memory
- train loss / val loss

## Local Scope

本地不下载 OpenWebText，也不要求加载真实数据集。本地只需要：

- 用 tiny `.txt` 或 `.jsonl` 训练一个小 vocab tokenizer。
- tokenize 成 `.npy`。
- 验证 encode/decode、EOT、dtype 和 token id 范围。
- 保持训练脚本能继续用 `np.load(..., mmap_mode="r")` 读取 tokenized `.npy`。
