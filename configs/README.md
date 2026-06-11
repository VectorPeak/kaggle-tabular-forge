# Configs

配置分两类：

- `*.example.yaml`：结构契约和人类意图示例。
- `experiments/*.example.yaml`：P2 可运行实验入口，可以传给 `ktab run --config`。

P2 当前可运行示例：

```powershell
uv run ktab run --config configs\experiments\p02_churn_logistic_basic.example.yaml
```

LightGBM 示例需要先安装可选依赖：

```powershell
uv sync --group dev --extra lgbm
uv run ktab run --config configs\experiments\p02_churn_lgbm_basic.example.yaml
```

命名约定：只有阶段性可运行示例配置使用 `pXX` 前缀，schemas 和全局契约配置保持语义命名。

P3 ensemble 示例：

```powershell
uv run ktab ensemble --config configs\ensembles\p03_candidate_ensemble.example.yaml
```

注意：P3 ensemble 配置会读取已有 registry 中的 completed parent experiments。示例里的 parent `experiment_id` 需要已经存在并具备对齐的 OOF 与 submission artifact。

P4 candidate factory 示例：

```powershell
uv run ktab factory --config configs\matrices\p04_churn_candidate_factory.example.yaml --dry-run
```

P4 会从 matrix 配置展开多个 P2 experiment config。建议先 dry-run，确认 `experiment_id`、`max_runs` 和 artifact 路径后再执行。

注意：示例里的 `experiment_id` 是固定的。重复运行同一个配置会因为 artifact 不允许覆盖而失败；需要重新运行时，请复制配置并改一个新的 `experiment_id`。

P6.1 EDA scan 示例：

```powershell
uv run ktab eda scan --config configs\eda\p06_churn_eda.example.yaml
```

这个命令会从标准 `train.csv` / `test.csv` / `sample_submission.csv` 布局读取数据，并写出：

- `eda_manifest.json`
- `eda_summary.md`
- `leakage_watchlist.json`
- `feature_backlog.json`

默认 artifact 目录：

```text
artifacts/eda_findings/<competition>/<eda_id>/
```
