# kaggle-tabular-forge Agent 指南

这是 `kaggle-tabular-forge` 的渐进式披露入口文档。

所有 Agent 进入仓库后先读这个文件。只有当任务需要更深细节时，再顺着链接进入 `docs/`。

## 0x01. 一页原则

`kaggle-tabular-forge` 是一个目标驱动的 Kaggle 表格赛实验工厂。

核心原则：

> 人类负责战略方向。LLM Agent 负责提案、审查、拆解和记录。`ktab` 负责执行。OOF、日志和可复现 artifact 决定什么是真结果。

本项目受到 NVIDIA KGMON Playbook、S6E3 第一名方案和 NVIDIA 关于生成式 AI 辅助 Kaggle 夺冠工作流的启发：大规模 EDA、大量特征工程、大候选池、严格 OOF 纪律，以及从大池中筛出更小但更多样的最终集成。

它不能变成黑盒 AutoML 工具，也不能把 public leaderboard 当成主优化循环。

## 0x02. 当前阶段与路线

当前仓库已经完成 P0-P5 地基：

1. `P0` 项目基座、规则、验收标准。
2. `P1` 实验证据流水线。
3. `P2` 配置驱动实验工作台。
4. `P3` 候选池与轻量 ensemble。
5. `P4` 候选实验工厂。
6. `P5` stacking / selection 基础能力：配置约束、候选筛选、hill climbing、OOF-safe 一层 stacking。

接下来的“重武器”路线按两期推进：

1. `P6`：EDA 驱动特征工程工作台 + 运行时底座  
   重点是可执行 EDA、fold-safe 特征工程、运行时后端抽象、预测载体抽象、registry 后端收口。
2. `P7`：pseudo-label + extra training + final candidate  
   重点是 teacher provenance、半监督风险控制、final retrain、submission review pack。

如果任务描述有歧义，优先改进契约、文档、示例和小型可验证脚手架，不要急着堆更多训练代码。

## 0x03. 不可违反的规则

这些规则优先级高于所有下层文档：

1. 不要把 public leaderboard 反馈当成主要优化循环。
2. 没有明确的人类批准，不要自动提交 Kaggle。
3. 没有 OOF/CV 证据，不要声称有改进。
4. 生成 OOF 前，不要在 full train 上训练目标相关变换。
5. Target encoding 必须 OOF-safe，或在 fold 内做 nested 编码。
6. 每个 ensemble、stacking、pseudo-label teacher candidate 都必须有对齐的 OOF 预测。
7. 不要覆盖旧实验 artifact；创建新的 `experiment_id`、`candidate_id` 或派生实验记录。
8. 失败、跳过和被拒绝的实验也必须记录。
9. 不要把凭证、Kaggle token、原始数据和生成 artifact 提交到 git。
10. LLM 输出在经过代码、OOF、日志和 artifact 验证前，只能视为假设。
11. 高风险流程必须职责隔离：`proposer != judge`，提案 Agent 不能自批。
12. `pseudo-label`、`external/original data`、`full-train extra training`、大预算扩容和最终 shortlist 冻结都需要人类审批。
13. 每个 search wave 必须先写预算、stop condition 和 promotion rule，再执行。
14. `extra training` 产物默认是 submission candidate，不是“CV improved experiment”。
15. 在 registry store 和 artifact commit 没有收口前，不要盲目并发写入。
16. 如果 GPU 路径无法给出与 CPU 路径等价的审计产物，不允许直接晋升为默认主路径。

## 0x04. 人类审批关口

以下动作必须显式获得人类批准，并把批准范围记录到 goal、report 或 review 中：

1. 真实 Kaggle submission。
2. 引入 pseudo-label。
3. 使用 original data、external data 或 reference data。
4. 进行 full-train extra training 或 final retrain。
5. 启动大预算 search wave，例如长时运行、多卡、批量扩容。
6. 冻结 final shortlist，进入最终候选包。

审批记录至少包含：

- 批准人
- 批准对象
- 预算范围
- 风险说明
- 回退条件

## 0x05. 进入仓库后的读取顺序

进入仓库后按顺序阅读：

1. `AGENTS.md`
2. `README.md`
3. `configs/` 与示例配置
4. `competitions/<competition>/goal.yaml`，如果存在
5. `docs/` 下与当前任务直接相关的文档
6. 最近的 experiment registry、candidate report 或 artifact manifest

不要默认阅读所有深层文档。根据任务选择。

## 0x06. 任务路由

行动前先查这张表：

| 任务类型 | 下一步阅读 | 主要输出 |
| --- | --- | --- |
| 新 `/goal` 或验收标准 | `docs/goal_contract.md` | 解析后的 goal 与验收记录 |
| 多 Agent 工作流 | `docs/agent_protocol.md` | 结构化任务图与职责分工 |
| Agent-native 产品集成 | `docs/agent_integration.md` | Agent/CLI/MCP 协作边界 |
| P0 项目基座 | `docs/p00_project_foundation.md` | 项目边界、规则与验收标准 |
| P1 实验证据流水线 | `docs/p01_evidence_pipeline.md` | OOF、submission、registry 证据链 |
| P2 配置实验工作台 | `docs/p02_experiment_workbench.md` | config-driven run 与 compare |
| P3 候选池与 ensemble | `docs/p03_candidate_ensemble.md` | OOF-backed candidate pool 与融合方案 |
| P4 候选工厂 / search wave | `docs/p04_candidate_factory.md` 和 `docs/artifact_contract.md` | matrix、run summary、candidate lineage |
| KGMON/S6E3 方法论 | `docs/kgmon_methodology.md` | proposal、phase plan 或 review |
| EDA / synthetic 关系 / 泄漏扫查 | `docs/goal_contract.md`、`docs/leakage_rules.md` | EDA 报告、watchlist、feature backlog |
| 特征工程 | `docs/feature_catalog.md` 和 `docs/leakage_rules.md` | feature proposal、risk label、config |
| 模型族 / GPU 路线 | `docs/model_zoo.md` 和 `docs/environment_notes.md` | model config、backend 建议、依赖决策 |
| 运行时后端 / 预测载体 / registry store | `docs/environment_notes.md` 和 `docs/artifact_contract.md` | runtime plan、format 决策、artifact contract |
| Hill climbing / stacking | `docs/stacking.md` 和 `docs/artifact_contract.md` | candidate selection、stacking plan、lineage |
| Pseudo-label / extra training | `docs/kgmon_methodology.md`、`docs/leakage_rules.md`、`docs/goal_contract.md` | 风险审查、teacher policy、promotion gate |
| 路径、输出、manifest、lineage | `docs/artifact_contract.md` | artifact contract 或 registry 变更 |
| 测试策略或测试数据集 | `docs/testing_strategy.md` | 测试计划、smoke 验收、阶段验证 |

## 0x07. 默认 `/goal` 阶梯

允许的高层 objective 至少包括：

- `eda`
- `build_baseline`
- `feature_test`
- `candidate_factory`
- `hill_climb`
- `stacking`
- `pseudo_label`
- `extra_training`
- `submit`
- `debug_gap`

默认执行阶梯：

1. Static review
2. Smoke test
3. Single fold
4. Full CV
5. Candidate factory / repeated seeds / 多样性扩展
6. Hill climbing / first-level stacking
7. Pseudo-label / extra training
8. Human-approved submission

允许跳步，但必须在 plan 中写清：

- 为什么跳步
- 跳过了哪些安全门
- 用什么证据替代

## 0x08. 搜索波次与晋升规则

每个 search wave 在执行前都要记录：

- `wave_id`
- `objective`
- `competition`
- `candidate source`
- `budget`
- `max_experiments`
- `max_parallel_runs`
- `stop_condition`
- `promotion_rule`
- `rejection_rule`

每个 candidate 至少要有：

- `candidate_id`
- `experiment_id`
- `parent_lineage`
- `feature_family`
- `model_family`
- `seed`
- `metric`
- `oof_path` 与 checksum
- `prediction_path` 与 checksum
- `decision_status`
- `decision_reason`

默认晋升门：

1. `baseline -> factory candidate`：必须 full CV completed，且 artifact 完整。
2. `factory candidate -> ensemble pool`：必须 OOF 对齐、fold 稳定、lineage 可追踪。
3. `ensemble pool -> hill climbing`：必须有 candidate report、OOF correlation 或多样性依据。
4. `hill climbing -> stacker input`：必须 OOF-safe，不允许 in-sample base predictions。
5. `teacher -> pseudo-label`：必须人类批准，且 teacher 是 OOF-backed 候选。
6. `approved recipe -> extra training`：必须冻结 lineage，不再临时改动特征和训练口径。

默认淘汰也要保留原因，不允许“静默失败”。

## 0x09. P5-P7 重武器路线

优先级顺序：`P5 > P6 > P7`

### P5：EDA 驱动特征工程工作台 + 运行时底座

目标：

- 让 `EDA -> feature proposal -> fold-safe feature build -> OOF compare` 闭环可执行。
- 把 CPU/GPU 后端、预测载体、registry store 抽象成一等公民。

关键能力：

- 结构化 EDA 报告、leakage watchlist、feature backlog。
- `frequency/count`、binning、interactions、nested target encoding。
- `ExecutionBackend`/`ModelAdapter` 抽象。
- `parquet + npy` 双预测载体。
- `csv|sqlite` 级别的 registry backend 规划。
- 可控并发的 factory runtime。

### P6：候选筛选 + hill climbing + 一层 stacking

目标：

- 从“轻量 averaging”升级到“OOF 驱动的选择与集成”。

关键能力：

- OOF correlation 报告。
- candidate diversity 报告。
- greedy hill climbing。
- logistic/ridge stacker。
- Level-1 stacker 的 OOF/test predictions、manifest、registry row、lineage。

### P7：pseudo-label + extra training + final candidate

目标：

- 把半监督和 final retrain 纳入严格 provenance。

关键能力：

- pseudo-label teacher policy。
- 阈值、样本量、混合比例和风险标签记录。
- extra training lineage 冻结。
- final shortlist review pack。
- human-approved submission candidate pipeline。

## 0x0A. 证据语言

没有 OOF 证据时，不要说：

- improved
- better
- strong baseline
- ready to submit
- solved overfitting

允许说：

- hypothesis to test
- smoke test passed
- single fold signal only
- full CV OOF improved by X
- candidate promoted with rationale
- CV/LB gap needs investigation
- submission candidate pending human approval

## 0x0B. 深层参考

需要时再读：

- `docs/README.md`: 文档索引与渐进式披露导航。
- `docs/goal_contract.md`: `/goal` 输入、自动发现规则、验收 schema、inconclusive 条件。
- `docs/p00_project_foundation.md`: P0 项目基座、规则、验收标准、非目标。
- `docs/p01_evidence_pipeline.md`: P1 实验证据流水线、smoke run、真实数据证据。
- `docs/p02_experiment_workbench.md`: P2 配置驱动实验工作台、model adapter、compare。
- `docs/p03_candidate_ensemble.md`: P3 候选池与轻量 ensemble。
- `docs/p04_candidate_factory.md`: P4 候选工厂、matrix 展开、run summary。
- `docs/agent_protocol.md`: AI Council 角色、Agent I/O、失败实验记录。
- `docs/agent_integration.md`: Agent/CLI/MCP 协作边界。
- `docs/kgmon_methodology.md`: S6E3 启发的方法论、MVP/增强范围、候选池思路。
- `docs/feature_catalog.md`: 特征家族与 feature registry 契约。
- `docs/model_zoo.md`: 模型家族、可选依赖理念、model registry 指南。
- `docs/stacking.md`: OOF candidates、hill climbing、logistic stacking、多层集成。
- `docs/leakage_rules.md`: target encoding、external data、public LB、submission 规则。
- `docs/artifact_contract.md`: artifact 路径、manifest、预测文件、experiment id、lineage。
- `docs/environment_notes.md`: 依赖分组、Windows/WSL/Kaggle/RAPIDS 边界。
- `docs/testing_strategy.md`: 测试层级、数据集、阶段验收。

## Prime Directive

大胆思考，谨慎实现，本地验证，诚实记录。

任何没有 CV/OOF、日志和可复现命令支撑的内容，都只是一个假设，不是结果。
