# 50M-token pilot 实验结果

本文记录 `llm_lab` 四个 100M-scale 架构在同一 tokenizer、数据集和训练脚本下的
第一轮短预算预训练结果。该实验的目的不是获得最终最优 loss，而是验证训练管线、
初始化修复后的 loss 尺度，以及四类架构在小 token budget 下的相对表现。

## 实验设置

运行环境：

- 平台：AutoDL 单卡 GPU 实例
- 训练入口：`scripts/run_experiment.py`
- 训练脚本：`scripts/train.py`
- 主要代码版本：
  - `modern_decoder`: `82a0f7d`
  - `original_transformer`, `qwen36`, `deepseek_v4`: `819ef5d`

数据与 tokenizer：

- tokenizer：byte-level BPE，`vocab_size = 16384`
- tokenizer 路径：`data/tokenizers/bpe_16k`
- 训练 tokens：`data/tokens/bpe_16k/train.npy`
- 验证 tokens：`data/tokens/bpe_16k/val.npy`
- `train.npy`: 49,206,260 tokens, `uint16`
- `val.npy`: 2,462,368 tokens, `uint16`

训练设置：

- `context_length = 1024`
- `batch_size = 8`
- `grad_accum_steps = 1`
- 每 step token 数：`8 * 1024 = 8192`
- 总步数：`6000`
- 总 token budget：`6000 * 8192 = 49,152,000`
- 学习率：`3e-4`
- 最小学习率：`3e-5`
- warmup：`600` steps
- cosine decay：`6000` steps
- 训练 dtype：`bfloat16`

本轮训练使用随机采样窗口，所以不是严格顺序遍历数据；但总 token budget 与
`train.npy` token 数基本相当，约等于一遍训练数据规模。

## 初始化修复

早期实验中，`TokenEmbedding` 使用 `std = 1.0` 初始化，并且模型默认使用 tied
embedding 作为输出头。这会让初始 logits 过大，导致 16k vocab 下随机初始化 loss
达到 100-200 量级。

本轮实验已修复为：

```text
embedding std = 1 / sqrt(d_model)
```

修复后，各模型 step 1 的 loss 均在 `10.1` 左右，接近随机 16k 分类的理论量级：

```text
ln(16384) = 9.704
```

这说明本轮结果可以作为有效 pilot 结果记录。

## 结果

`val_ppl` 按 `exp(val_loss)` 计算。由于 tokenizer、数据集和切分方式会影响 PPL，
该数值仅用于本项目内横向比较，不应直接与其他数据集或作业设定比较。

| 模型 | 架构 | 总参数 | 每 token 激活参数 | train loss | val loss | val PPL | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modern_decoder_100m` | `modern_decoder` | 100,685,568 | 100,685,568 | 4.4012 | 4.7360 | 114.0 | 49,406 |
| `qwen36_100m` | `qwen36` | 102,407,760 | 102,407,760 | 4.6202 | 4.9291 | 138.3 | 42,385 |
| `deepseek_v4_100m` | `deepseek_v4` | 105,554,600 | 58,368,680 | 4.6468 | 4.9696 | 144.0 | 67,645 |
| `original_transformer_100m` | `original_transformer` | 97,555,968 | 97,555,968 | 7.2069 | 7.5161 | 1837.4 | 62,938 |

## 初步观察

`modern_decoder` 是本轮 50M-token pilot 中 loss 最低的模型。它使用 RoPE、
RMSNorm、SwiGLU、GQA、pre-norm 和 QK norm 等现代 dense decoder 组件，在短预算
训练下表现最稳定。

`qwen36` 的验证 loss 略高于 `modern_decoder`，但仍显著优于原始 Transformer。
当前实现采用 3 个 linear mixer + 1 个 full attention 的周期结构；在 1024 context
和 50M token budget 下，full attention dense decoder 更容易占优。

`deepseek_v4` 的验证 loss 与 `qwen36` 接近，略差一些，但吞吐最高。它的总参数约
105.6M，但每 token 激活参数约 58.4M，MoE/稀疏激活带来了明显速度优势。

`original_transformer` 速度较高，但 loss 明显落后。这个结果符合预期：post-norm、
sinusoidal position、ReLU FFN 和标准 MHA 在当前小规模预训练设置下不如现代 decoder。

## 限制

本轮实验是 pilot，不是最终 benchmark。

- 数据规模较小：训练集约 49.2M tokens，本轮 6000 step 约消耗 49.15M tokens。
- 数据来自 OpenWebText 子集，仍可能包含网页噪声、编码异常和重复内容。
- 当前只比较了短上下文 `context_length = 1024`，尚未测试长上下文行为。
- 当前只记录 next-token validation loss，尚未做生成质量、长上下文、记忆能力或
  下游任务评估。
- `qwen36` 与 `deepseek_v4` 是公开架构思想的 100M-scale 近似复现，不包含官方
  production kernel、完整训练 recipe 或真实大模型规模细节。
- 本轮 `qwen36` 和 `deepseek_v4` 结果来自 MTP loss 接入训练脚本之前，因此虽然
  配置中存在 `mtp_layers = 1`，MTP head 没有贡献本轮 next-token loss。
- checkpoint 清理后，`checkpoint_count` 不再能反映完整训练过程中曾保存的所有
  checkpoint 数量。

## 下一步

短期建议：

1. 给 `scripts/summarize_runs.py` 增加 `val_ppl` 列。
2. 继续保留 6000-step pilot 配置，作为快速回归和架构 sanity check。
3. 重新运行 `qwen36` / `deepseek_v4` 的 6000-step pilot，比较接入 MTP loss 前后的
   next-token loss 与训练速度。
4. 为 60000-step 长训准备更大的 OpenWebText token 文件，避免 500M token budget
   反复采样同一 49M-token 子集。

更完整的 benchmark 应包含：

- 500M-token budget 长训结果
- 更大 OpenWebText 子集
- PPL 曲线和 tokens/s 曲线
- 长上下文评估
- 固定 prompt 的生成样例对比
