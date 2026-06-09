# P3 候选池与轻量 Ensemble

P3 规划并实现候选池和轻量 ensemble 阶段。当前 MVP 已经包含 candidate pool、alignment gate、simple/rank/weighted average、ensemble artifact 写入、registry lineage 和 `ktab ensemble --config` 入口。

本文仍然保留设计边界：P3 MVP 不做 public-LB-driven weight search、hill climbing、stacking、HPO 或自动 Kaggle submission。

## 目标

P3 要把已经完成的 P2 实验转成一个小而可信的 candidate pool，然后用透明的融合方法评估结果。

本阶段要回答：

- 哪些 completed experiment 可以成为 ensemble candidate？
- 它们的 OOF 和 test prediction 是否和同一批行、同一个 target 语义对齐？
- 使用了哪一种 averaging recipe？
- 这个 ensemble 来自哪些 parent experiment？
- ensemble 最终是接受、拒绝，还是推迟，原因是什么？

## Candidate Pool

候选池从 experiment registry 读取，默认只接受：

- `status: completed`。
- OOF path 和 test prediction path 不为空。
- `competition`、`target`、metric、prediction type 兼容。
- CV protocol 兼容，或者有明确兼容性说明。
- 泄漏风险为 low，或已经完成审查。
- 模型、特征、seed、preset 至少有一个维度形成有效差异，避免重复候选。

P3 的默认候选池应该保持小而可解释。这个阶段优先建立可信比较，而不是追求大规模搜索。

## 对齐过滤

任何 averaging 之前，P3 必须检查：

- OOF 行数等于 train 行数。
- test prediction 行数等于 sample submission 行数。
- OOF 的 id、fold、target 对齐同一批 train rows。
- test id 与 sample submission 顺序对齐，或可以安全按 id join。
- prediction type 兼容：probability 与 probability 融合，rank 与 rank 融合；如需转换，必须显式记录。
- 不允许候选使用 public leaderboard 反馈来决定 weights、thresholds、features 或训练逻辑。

对齐失败时，必须阻断 ensemble artifact 生成。

## Ensemble 方法

P3 从透明方法开始：

- `simple_average`：对齐后的概率预测做算术平均。
- `rank_average`：当模型校准差异较大时，对每个候选的预测 rank 做平均。
- `weighted_average`：使用配置中写死的权重；权重来自 OOF 证据或事前记录的先验，不能来自 public LB 反馈。

weight search、hill climbing、stacking 和 meta-model training 都延后，除非后续阶段补上更强的安全门。

## 计划 Artifact

P3 ensemble MVP 写出：

- `config.yaml`：parent experiment IDs 和 recipe 参数。
- `run_manifest.json`：命令、时间、环境、输入路径。
- `metrics.json`：OOF ensemble score。
- `fold_metrics.csv`：fold 级 ensemble metric。
- `oof.parquet` 或 `oof.npy` 及其 schema。
- `submission.csv` 或 `test_predictions.parquet`。
- `ensemble_manifest.json`：recipe、parents、alignment checks、hashes。
- `candidate_pool.parquet`：通过过滤的候选池。
- `selection_report.md`：选择、拒绝和风险说明。

artifact 继续使用 P1/P2 的 artifact root 约定，并记录 parent lineage。

## Registry

ensemble registry row 应该标明它是派生实验：

- `experiment_id`。
- `competition`。
- `goal_id`。
- `cv_protocol_id`。
- `model_family: ensemble`。
- `model_preset` 或 recipe name。
- `prediction_type`。
- parent experiment IDs。
- OOF score 与 fold summary。
- artifact paths。
- leakage risk 与 review status。
- decision status 与 reason。

registry compare 应该让 ensemble 能和 base model 一起比较，同时让 lineage 明确可查。

## 验收

P3 可验收时，系统应该能够：

- 从 completed registry rows 构建 candidate pool。
- 阻断 OOF 或 test prediction 未对齐的候选。
- 对齐后运行 simple、rank、fixed weighted average。
- 写出 ensemble artifact 和 parent lineage。
- 新增 registry row，且不覆盖 parent experiment。
- 报告 OOF score 和 decision status。
- 没有人类明确批准时，不执行真实 Kaggle submission。

## 延后项

P3 暂缓：

- public-LB-driven weight search。
- 自动 Kaggle submission。
- 多层 stacking。
- meta-model training。
- ensemble weights HPO。
- 缺少验证计划的复杂 calibration blending。
- 使用 original/external data 做 ensemble 决策。
