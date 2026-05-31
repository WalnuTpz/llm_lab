# Experiment Protocol

本项目比较四个约 100M-scale LLM 架构：

- `original_transformer`
- `modern_decoder`
- `qwen36`
- `deepseek_v4`

实验目标是比较架构思想和模块接口，不比较官方权重、官方训练数据、post-training
或工业 kernel。正式训练预期在云端 GPU 环境执行，本地只做 CPU smoke tests。

## Fixed Inputs

四个模型必须共享：

- 同一个 16k byte-level BPE tokenizer。
- 同一份 OpenWebText tokenized train/val split。
- 同一个 context length。
- 同一个 token budget。
- 同一个 optimizer family 和 learning-rate schedule。
- 同一个 seed，除非实验显式记录多 seed。

正式配置统一指向：

```yaml
data:
  train_data: data/tokens/bpe_16k/train.npy
  val_data: data/tokens/bpe_16k/val.npy
  tokenizer_path: data/tokenizers/bpe_16k
  dataset_dtype: uint16
```

本地不会下载或加载 OpenWebText；云端训练前按
`docs/data_tokenizer_plan.md` 生成这些文件。

## Token Budget

所有模型按训练 token 数比较，不按 epoch 比较。

建议阶段：

```text
debug: 1M tokens
pilot: 50M tokens
main: 300M or 500M tokens
```

`main` token budget 要等 `modern_decoder` 在目标 GPU 上跑出实际 tokens/sec 后再
最终确定，避免超过单卡 4090 的 10 小时目标。

## Required Metrics

每次正式 run 至少记录：

- config name
- architecture
- git commit
- tokenizer path
- train data path
- val data path
- total parameters
- active parameters per token
- train tokens consumed
- batch size
- context length
- gradient accumulation, if used
- learning rate
- train loss
- val loss
- tokens/sec
- wall-clock seconds
- peak GPU memory, if CUDA is used

MoE 模型必须同时报告 `total_parameters` 和 `active_parameters_per_token`。
`deepseek_v4` 不能只按 total parameters 和 dense 模型比较。

## Logging

每个 run 使用自己的 `runtime.run_dir`，推荐结构：

```text
runs/<architecture>/
  metrics.jsonl
  config.yaml or resolved_config.json
```

`metrics.jsonl` 每行是一条 JSON record，至少包含：

```json
{"type": "train", "step": 10, "loss": 4.2, "lr": 0.0003}
{"type": "eval", "step": 200, "val_loss": 4.0}
```

训练脚本可以打印人类可读日志，但可分析结果必须来自 JSONL。

## Checkpoints

每个 run 使用自己的 `runtime.checkpoint_dir`，推荐结构：

```text
checkpoints/<architecture>/
  step_00002000.pt
  latest.pt
```

checkpoint 必须包含：

- model state dict
- optimizer state dict
- completed step
- config dict
- random seed metadata, where available

恢复训练时从下一个 step 继续，不能重复已经完成的 step。

## Evaluation

val loss 使用同一个 `data.val_data`，按固定 `runtime.eval_batches` 随机抽 batch
估计。CPU smoke tests 可以把 eval batches 设得很小；正式云端 run 应使用足够稳定
的 eval batch 数。

评估间隔由 `runtime.eval_interval` 控制。建议：

```text
debug: every 20-50 steps
pilot/main: every 200-1000 steps
```

## Acceptance Before Cloud Training

云端训练前，本地必须通过：

- config loading tests
- model construction/forward/backward smoke tests
- tokenizer train/load/tokenize tests
- real `.npy` data train smoke test
- checkpoint save + resume smoke test

这些测试只使用 tiny fixture，不代表正式 loss 或性能。
