# LLM Lab

100M-scale LLM architecture reproduction lab.

This repository compares four language model architectures under one package,
one config system, one training/generation/benchmark interface, and one set of
CPU smoke tests:

- `original_transformer`
- `modern_decoder`
- `qwen36`
- `deepseek_v4`

The goal is architecture-level scaled reproduction, not official weights,
official data, post-training, or production CUDA kernels.

## Scope

The main experiment scale is around 100M parameters. The intended hardware for
real training is a single RTX 4090, with each model targeted to fit within a
10-hour training budget. This local development environment may not have a GPU,
so tests only require CPU smoke validation with small fixture configs.

See [docs/implementation_requirements.md](docs/implementation_requirements.md)
for the full implementation requirements and model positioning.
See [docs/experiment_protocol.md](docs/experiment_protocol.md) for the cloud
training comparison protocol.
See [docs/cloud_training.md](docs/cloud_training.md) for the end-to-end cloud
runbook.

## Models

`original_transformer` is a historical decoder-only baseline using original
Transformer-style components: full MHA, sinusoidal positions, LayerNorm, and
ReLU FFN.

`modern_decoder` is the stable modern dense baseline: RMSNorm, RoPE, SwiGLU,
GQA, and optional QK-Norm.

`qwen36` is a scaled reproduction of the public Qwen3.6 topology. It preserves
the 3:1 linear-mixer/full-attention schedule with Gated DeltaNet-like and Gated
Attention-like blocks.

`deepseek_v4` is a scaled reproduction of the public DeepSeek V4 topology. It
keeps MoE, shared expert support, MTP heads, partial RoPE, small KV-head
attention, and CSA/HCA-like compressed attention interfaces.

Architecture notes are in:

- [docs/model_notes/original_transformer.md](docs/model_notes/original_transformer.md)
- [docs/model_notes/modern_decoder.md](docs/model_notes/modern_decoder.md)
- [docs/model_notes/qwen36.md](docs/model_notes/qwen36.md)
- [docs/model_notes/deepseek_v4.md](docs/model_notes/deepseek_v4.md)

## Layout

```text
configs/                    # 100M-scale experiment configs
docs/                       # requirements and architecture notes
scripts/                    # CLI entrypoints
src/llm_lab/                # Python package
tests/                      # CPU smoke tests and small fixture configs
```

Runtime outputs are ignored by Git:

```text
data/
artifacts/
checkpoints/
runs/
wandb/
```

## Environment

The project uses `uv`.

```bash
uv sync
```

If `uv` has trouble on a Windows UNC path, run commands from WSL or use the
existing `.venv` created by `uv` inside the WSL workspace.

## Test

```bash
uv run pytest
```

The expected test set is CPU-only and uses small configs in
`tests/fixtures/configs/`. It verifies config loading, model construction,
forward pass, loss, backward pass, tokenizer behavior, and parameter accounting.

## Inspect A Model

```bash
uv run python scripts/inspect_model.py --config configs/modern_decoder.yaml --device cpu
```

For the 100M configs, `--device cpu` is useful for parameter inspection only;
real training is intended for GPU.

To compare all formal configs without allocating model weights:

```bash
uv run python scripts/inspect_all_models.py
```

## Smoke Train

```bash
uv run python scripts/train.py \
  --config tests/fixtures/configs/modern_decoder_small.yaml \
  --smoke \
  --device cpu \
  --max-iters 1
```

Smoke training uses random token IDs and does not require a dataset.
Training writes `metrics.jsonl` under `runtime.run_dir` and checkpoints under
`runtime.checkpoint_dir`; resume with `--resume` to load `latest.pt`.

## Tokenize Text

```bash
uv run python scripts/tokenize_text.py \
  --input data/raw.txt \
  --output data/tokens.npy \
  --add-eot
```

The current tokenizer pipeline includes a minimal byte-level tokenizer. It is
intended as a stable unified baseline for smoke tests and early experiments.

Formal experiments use an OpenWebText-trained 16k byte-level BPE tokenizer.
See [docs/data_tokenizer_plan.md](docs/data_tokenizer_plan.md).

## Generate

```bash
uv run python scripts/generate.py \
  --config configs/modern_decoder.yaml \
  --prompt "hello" \
  --max-new-tokens 16 \
  --device cpu
```

When passing comma-separated token IDs from PowerShell, quote the value:

```bash
uv run python scripts/generate.py --config tests/fixtures/configs/modern_decoder_small.yaml --prompt-ids "1,2"
```

## Benchmark

```bash
uv run python scripts/benchmark.py \
  --config tests/fixtures/configs/qwen36_small.yaml \
  --device cpu \
  --steps 3
```

GPU runs additionally report CUDA peak memory.
