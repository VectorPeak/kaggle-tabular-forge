# 测试策略

本文档记录 `kaggle-tabular-forge` 的测试层级和默认测试数据集。

测试目标不是“尽快跑高分”，而是验证 P0 实验证据流水线是否可信：配置能解析，数据能审计，CV 能复现，OOF 能对齐，artifact 能追踪，安全门能阻断不可信结果。

## 默认测试数据集

默认使用 Kaggle 客户流失预测数据集：

```yaml
competition: playground-series-s6e3
kaggle_slug: playground-series-s6e3
task_type: binary_classification
target: Churn
metric: roc_auc
metric_mode: maximize
id_column: id
```

本项目不会提交原始数据。数据应放在本地：

```text
data/playground-series-s6e3/raw/train.csv
data/playground-series-s6e3/raw/test.csv
data/playground-series-s6e3/raw/sample_submission.csv
```

如果使用 original/reference data，放在：

```text
data/playground-series-s6e3/external/original.csv
```

任何 external/original data 都必须经过 leakage review。

## 测试层级

### 1. 文档与配置测试

目标：确保仓库入口和配置契约可读、可解析。

应检查：

- Markdown 代码块闭合。
- README 和 AGENTS 中的 `.md` 链接存在。
- JSON schema 能解析。
- YAML example 能解析。
- 没有乱码、冲突标记或占位标记。

### 2. Config Schema 测试

目标：确保用户写错配置时能尽早失败。

应覆盖：

- `configs/goal.example.yaml`
- `configs/competition.example.yaml`
- `configs/experiment.example.yaml`
- `configs/paths.example.yaml`
- `configs/agents.example.yaml`
- `configs/feature_catalog.example.yaml`
- `configs/model_zoo.example.yaml`
- `configs/stacking.example.yaml`

后续 CLI 应提供：

```text
ktab validate-config --config configs/competition.example.yaml
```

### 3. 数据审计测试

目标：验证客户流失数据能被正确识别和审计。

应检查：

- train/test/sample submission 文件存在。
- `id` 列存在且可对齐。
- `Churn` 目标列存在。
- sample submission 的目标列名称正确。
- train/test 行数和列数被记录。
- 缺失值、类别基数、目标分布被记录。
- train/test drift 初步报告能生成。

### 4. CV/OOF 契约测试

目标：验证 OOF 证据链可以建立。

应检查：

- `cv_protocol_id` 被生成或读取。
- StratifiedKFold 能基于 `Churn` 生成。
- fold assignment 可复现。
- 每一行 train 只属于一个 validation fold。
- OOF 输出能按 `id` 回填到原始 train。
- OOF schema 记录 row count、id column、target column、prediction column、fold column、checksum。

### 5. Baseline Smoke 测试

目标：验证最小训练闭环。

第一版可只跑小样本或 single fold：

- logistic/ridge baseline
- 或 LightGBM/XGBoost/CatBoost 之一

必须产出：

- run command
- config snapshot
- fold metrics
- OOF prediction
- test prediction
- model manifest
- environment manifest
- git manifest
- validation report

没有 OOF artifact 的 smoke 只能证明代码能跑，不能证明模型有效。

### 6. Artifact 与 Registry 测试

目标：确保实验可以进入候选池。

应检查：

- `experiment_id` 唯一。
- 旧 artifact 不被覆盖。
- `metrics.json` 可读。
- `fold_metrics.csv` 可读。
- OOF/test prediction 路径存在。
- leakage risk 和 status 被记录。
- failed/rejected/inconclusive 实验也有记录。

### 7. Ensemble Dry Run

目标：验证 ensemble 只消费可信候选。

应检查：

- 只使用有 aligned OOF 的候选。
- 候选来自 registry，而不是随便扫描文件。
- simple average / rank average 可基于 OOF 计算。
- hill climbing 只看 OOF，不看 public leaderboard。
- selection log 记录 accepted/rejected reason。

### 8. Submission Review Dry Run

目标：验证提交前安全门。

应检查：

- submission.csv 与 sample submission 格式一致。
- final prediction 可追溯到 experiment lineage。
- local OOF/CV 证据存在。
- 无 leakage blocker。
- `human_approved` 明确为 true 才允许真正提交。

## 不纳入 P0 的测试

P0 不测试：

- 完整 HPO。
- 多 GPU 训练。
- 全量神经网络 model zoo。
- 多层 stacking 生产流程。
- 自动 Kaggle submission。
- public leaderboard 驱动的权重搜索。

这些属于后续阶段。

## 验收标准

测试体系 P0 完成时应满足：

- 文档和配置能解析。
- 客户流失数据集路径约定清楚。
- 数据不进入 git。
- 能定义 `cv_protocol_id`。
- 能描述 OOF/test prediction 对齐要求。
- 能说明 baseline smoke 需要哪些 artifact。
- 能阻止没有 OOF 证据的 improvement 声明。
