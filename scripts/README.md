`regenerate_train_data.py` 最近做了较大改动，除了原来的数据重生成模式外，现在还支持 `entropy` / `sample NLL` 分析。

## 1. 普通数据重生成

用途：调用一个或多个 SGLang server，对输入 jsonl 中的对话重新生成 assistant 回复，并写回新的 jsonl。

常用参数：

```bash
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --concurrency 128 \
  --max-tokens 4096 \
  --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 \
  --temperature 0.8 \
  --input-file-path /path/to/input.jsonl \
  --output-file-path /path/to/output.jsonl \
  --resume \
  --is-reasoning-model
```

说明：

- `--output-file-path`：普通重生成模式下必须提供。
- `--resume`：若输出文件已存在，则跳过已经处理过的样本。
- `--no-think`：对支持 thinking 的模型关闭 thinking。
- `--is-reasoning-model`：如果返回中有 `reasoning_content`，会一并写入输出。
- `--is-gpt-oss`：对 GPT-OSS 随机设置 `reasoning_effort`。
- `--top-p` / `--top-k` / `--repetition-penalty`：可选采样参数。

## 2. Entropy 模式

用途：不写重生成结果，只统计目标模型在数据集上的平均 token loss / entropy。

示例：

```bash
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --concurrency 128 \
  --max-tokens 4096 \
  --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 \
  --temperature 0 \
  --input-file-path /path/to/input.jsonl \
  --entropy
```

说明：

- `--entropy` 打开后，不再要求 `--output-file-path`。
- 最终会打印：
  - 成功 / 失败样本数
  - 总生成 token 数
  - 平均每样本 token 数
  - 平均 token loss / cross-entropy（nats / bits）

## 3. Sample NLL 曲线模式

用途：在 `entropy` 模式下，从数据集中抽样若干条样本，保存每条样本的 token-level NLL 曲线，并汇总均值曲线。

示例：

```bash
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --concurrency 128 \
  --max-tokens 4096 \
  --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 \
  --temperature 0 \
  --input-file-path /path/to/input.jsonl \
  --entropy \
  --sample-num 10000 \
  --fig-dir /mnt/shared-storage-user/leihaodi/imo/SpecForge/entropy-try/nemotron-stem-40960-16000maxtoken-off
```

会输出：

- 每个被抽中的样本：
  - `{line_number}.png`：该样本的 token-level NLL 曲线
  - `{line_number}.txt`：该样本的对话内容（含 reasoning_content，如有）
- 聚合结果：
  - `mean.png`：按 token 位置求平均后的均值 NLL 曲线
  - `mean_nll.json`：均值 NLL 数据

说明：

- `--sample-num` 只能和 `--entropy` 一起用。
- 设置 `--sample-num` 时，必须同时提供 `--fig-dir`。
- 抽样是**确定性的**：由 `input_file_path`、可用行数和 `sample_num` 共同决定。
  - 这意味着同一个数据集在 think / no-think 两次运行时，如果其余条件一致，会抽到相同的样本，方便对比。

## 4. 只保留均值图，不保存逐样本图

如果只想统计 sampled mean curve，不想保存每条样本的 png / txt，可以加：

```bash
--ignore-fig
```

示例：

```bash
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3.5-35B-A3B \
  --concurrency 128 \
  --max-tokens 4096 \
  --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 \
  --temperature 0 \
  --input-file-path /path/to/input.jsonl \
  --entropy \
  --sample-num 10000 \
  --fig-dir /mnt/shared-storage-user/leihaodi/imo/SpecForge/entropy-try/nemotron-stem-40960-16000maxtoken-off \
  --ignore-fig
```

此时只会输出：

- `mean.png`
- `mean_nll.json`

不会输出：

- `{line_number}.png`
- `{line_number}.txt`

## 5. 其它补充

- `--num-samples`(这是原来就有的参数，注意区别于我们自己加的`--sample-num`)：只处理输入文件前 `N` 条样本；可用于快速测试。
- 如果 `--entropy` 未开启，则必须提供 `--output-file-path`。
- 如果 `--sample-num` 开启，则要求：
  - `--sample-num > 0`
  - `--fig-dir` 非空



