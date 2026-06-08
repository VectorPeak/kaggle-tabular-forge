# P1 Evidence Pipeline

P1 的目标是交付最小可运行证据流水线：给定一个表格数据集，项目可以运行一条诚实 baseline，生成 OOF、submission、metrics、manifest 和 registry 记录。

P1 从 P0 的项目基座和证据契约出发，验证真实命令能跑通、artifact 能追踪、行数能对齐、registry 能记录。P1 不是冲榜阶段，也不是完整 model zoo。P1 的核心验收是：实验可以跑，证据可以查，失败可以被阻断。

## Scope

P1 包含：

- `pyproject.toml` 与 `uv` 开发环境。
- `ktab` CLI。
- YAML + JSON Schema 配置校验。
- tiny churn fixture。
- StratifiedKFold fold assignment。
- Logistic Regression OOF baseline。
- ROC AUC scoring。
- OOF parquet。
- submission CSV。
- metrics、fold metrics、feature manifest、model manifest、run manifest。
- experiment registry CSV。
- submission review dry run。

P1 不包含：

- 自动 Kaggle submit。
- public leaderboard 优化。
- LightGBM/XGBoost/CatBoost 默认模型。
- nested target encoding。
- original-data lookup。
- 多层 stacking。
- GPU/RAPIDS 栈。

## Architecture

```text
src/ktabforge/
├── cli.py                    # ktab 命令入口
├── config/                   # YAML 读取与 JSON Schema 校验
├── data/                     # train/test/sample_submission 读取与审计
├── cv/                       # StratifiedKFold fold 生成
├── features/                 # P1 最小安全特征选择
├── models/                   # Logistic Regression baseline
├── metrics/                  # ROC AUC 等指标
├── pipeline/                 # smoke evidence pipeline 编排
├── artifacts/                # artifact 路径、写入、manifest、对齐
├── registry/                 # experiment registry 写入
├── safety/                   # OOF/submission gate 与泄漏 review stub
└── utils/                    # hash、env、git、time、logging
```

## Commands

安装依赖：

```powershell
uv sync --group dev
```

运行测试：

```powershell
uv run pytest
uv run ruff check .
```

校验配置：

```powershell
uv run ktab validate-config `
  --config configs\competition.example.yaml `
  --schema configs\schemas\competition.schema.json
```

运行 tiny smoke：

```powershell
uv run ktab smoke `
  --data-dir tests\fixtures\data\churn_tiny `
  --artifact-root artifacts `
  --competition churn_tiny `
  --experiment-id p1-local-smoke `
  --target Churn `
  --id-column id `
  --n-splits 3 `
  --seed 42
```

## Acceptance

P1 可验收条件：

- `uv run pytest -q` 通过。
- `uv run ruff check .` 通过。
- `uv run ktab --help` 能展示 CLI。
- `uv run ktab validate-config` 能校验 competition example。
- `uv run ktab smoke` 能生成 artifact。
- OOF 行数等于 train 行数。
- submission 行数等于 test 行数。
- registry 记录 status 为 `completed`。
- 不执行真实 Kaggle submit。

## Current Verification

本阶段已经用 `tests/fixtures/data/churn_tiny` 验证最小链路：

```text
config/schema
  -> data load
  -> data audit
  -> stratified folds
  -> logistic OOF baseline
  -> metrics
  -> oof.parquet
  -> submission.csv
  -> experiment_registry.csv
```

## Real Data Smoke

2026-06-08 已使用本地真实 Kaggle 数据完成 P1 smoke：

```text
data/playground-series-s6e3/raw/train.csv
data/playground-series-s6e3/raw/test.csv
data/playground-series-s6e3/raw/sample_submission.csv
```

数据规模：

```text
train rows: 594,194
test rows: 254,655
sample submission rows: 254,655
```

运行命令：

```powershell
uv run ktab smoke `
  --data-dir data\playground-series-s6e3\raw `
  --artifact-root artifacts `
  --competition playground-series-s6e3 `
  --experiment-id p1-real-smoke-20260608145055 `
  --target Churn `
  --id-column id `
  --n-splits 5 `
  --seed 42
```

结果：

```text
status: completed
metric: roc_auc
oof_score: 0.9079518373824657
oof rows: 594,194
submission rows: 254,655
```

fold metrics：

```text
fold  roc_auc   rows
0     0.907478  118,839
1     0.908935  118,839
2     0.908105  118,839
3     0.909114  118,839
4     0.906162  118,838
```

生成 artifact：

```text
artifacts/experiments/playground-series-s6e3/p1-real-smoke-20260608145055/
artifacts/oof/playground-series-s6e3/p1-real-smoke-20260608145055/oof.parquet
artifacts/submissions/playground-series-s6e3/p1-real-smoke-20260608145055/submission.csv
artifacts/registry/playground-series-s6e3/experiment_registry.csv
```
