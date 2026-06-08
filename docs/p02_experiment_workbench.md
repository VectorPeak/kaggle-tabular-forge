# P2 配置驱动实验工作台

P2 的目标是把 P1 的一次性 smoke pipeline 升级为配置驱动、可复现、可比较的实验工作台。

P2 不追求完整 model zoo、复杂特征工程、HPO、stacking 或 Kaggle 自动提交。P2 的验收重点是：实验能由配置驱动，artifact 可追踪，registry 可比较，模型 adapter 可扩展。

## Multi-Agent Review Summary

本阶段启动了三个只读审核 Agent：

- 架构审核：建议 P2 聚焦 `ktab run --config`、统一 runner、model adapter、registry compare。
- 建模/泄漏审核：建议优先 LightGBM + basic/categorical/frequency 等低风险特征，target encoding 延后。
- 测试/交付审核：建议保留 P1 `ktab smoke`，新增 P2 `ktab run --config` 与 `ktab compare`，并用 tiny fixture 做默认 CI。

合并建议后，P2 执行范围收敛为：

- 配置驱动实验入口。
- config snapshot 与 config hash。
- 统一模型 adapter registry。
- Logistic Regression、LightGBM、XGBoost adapter。
- `basic_inferred` feature set resolver。
- registry 扩展字段。
- compare/report 基础能力。

延后到 P2.5/P3：

- CatBoost 默认验证。
- target encoding。
- original/external lookup。
- snap features。
- stacking / hill climbing。
- HPO。
- GPU/RAPIDS。
- Kaggle auto-submit。

## Architecture

```text
src/ktabforge/
├── config/
│   └── experiment.py             # ExperimentConfig 与 config hash
├── pipeline/
│   ├── runner.py                 # P2 config-driven 主流程
│   └── results.py                # ExperimentRunResult
├── models/
│   ├── base.py                   # 统一 OOF 模型结果与 sklearn helper
│   ├── logistic.py               # Logistic adapter
│   ├── lightgbm.py               # LightGBM adapter，延迟 import
│   ├── xgboost.py                # XGBoost adapter，延迟 import
│   └── registry.py               # model family dispatch
├── features/
│   └── sets.py                   # basic/basic_inferred feature set resolver
├── registry/
│   └── experiments.py            # 保留扩展字段的 registry API
└── reports/
    └── compare.py                # registry compare
```

## Commands

Logistic config-driven run：

```powershell
uv run ktab run --config configs\experiments\p02_churn_logistic_basic.example.yaml
```

LightGBM config-driven run：

```powershell
uv sync --group dev --extra lgbm
uv run ktab run --config configs\experiments\p02_churn_lgbm_basic.example.yaml
```

Compare：

```powershell
uv run ktab compare --artifact-root artifacts --competition playground-series-s6e3 --top-n 10
```

## Real Data Execution Results

命名迁移说明：下面的真实数据指标来自同参数 P2 配置的历史运行。本次只把示例配置和推荐 `experiment_id` 从 `p2` 口径整理为 `p02` 口径，没有重新生成真实数据 artifact。

数据：

```text
competition: playground-series-s6e3
target: Churn
id_column: id
train rows: 594,194
test rows: 254,655
```

P2 Logistic：

```text
recommended_experiment_id: p02-logistic-basic-example
model_family: logistic_regression
model_preset: baseline
feature_set: basic_inferred
n_splits: 5
oof_score: 0.9079518373824657
status: completed
```

P2 LightGBM：

```text
recommended_experiment_id: p02-lgbm-basic-example
model_family: lightgbm
model_preset: smoke
feature_set: basic_inferred
n_splits: 3
oof_score: 0.9137407561035117
status: completed
```

LightGBM fold metrics：

```text
fold  roc_auc   rows
0     0.913729  198,065
1     0.914546  198,065
2     0.912992  198,064
```

P2 compare 排序结果中，LightGBM 高于 Logistic：

```text
p02-lgbm-basic-example      roc_auc 0.9137407561035117
p02-logistic-basic-example  roc_auc 0.9079518373824657
```

## Acceptance

P2 当前验收：

- `uv run pytest -q` 通过。
- `uv run ruff check .` 通过。
- `ktab run --config` 能跑 tiny fixture 和真实数据。
- `ktab compare` 能读取 registry 并按 OOF 排序。
- 每个 P2 实验生成 `config.yaml` snapshot。
- registry 包含 `model_preset`、`feature_set`、`config_hash`、`feature_manifest_hash`。
- LightGBM optional dependency 通过 `uv sync --extra lgbm` 安装并验证。
- 旧的 P1 `ktab smoke` 仍然可用。

## Known Limits

- XGBoost adapter 已实现，但尚未在真实数据上执行验证。
- `frequency_count`、binning、interactions 尚未实现。
- compare 目前输出 CSV 文本，还不是完整 Markdown report。
- 没有 public leaderboard 逻辑，也不自动提交 Kaggle。
- LightGBM 运行时出现 sklearn feature-name warning，不影响 artifact 生成和行数对齐，但后续可以清理。
