# kaggle-tabular-forge Agent 指南

这是 `kaggle-tabular-forge` 的渐进式披露入口文档。

所有 Agent 进入仓库后先读这个文件。只有当任务需要更深细节时，再顺着链接进入 `docs/`。

## 0x01. 先读这里

`kaggle-tabular-forge` 是一个目标驱动的 Kaggle 表格赛实验工厂。

核心原则：

> 人类负责战略方向。LLM Agent 扩展实验搜索空间。GPU 加速执行。OOF、日志和可复现 artifact 决定什么是真结果。

本项目受到 NVIDIA/KGMON S6E3 这类高吞吐表格赛工作流启发：大量特征想法、大量候选模型、严格 OOF 纪律，以及从大候选池中筛出更小但更多样的最终集成。

它不能变成黑盒 AutoML 工具。

## 0x02. 当前工作模式

仓库仍处于地基阶段。

当前优先级：

1. 稳定目录结构。
2. 稳定配置与 schema 契约。
3. 稳定 artifact 与 OOF 契约。
4. 清楚记录 Agent 工作流。
5. 在契约稳定前，避免实现庞大的训练框架。

如果任务描述有歧义，优先改进契约、文档、示例和小型可验证脚手架，不要急着构建昂贵的训练机器。

## 0x03. 不可违反的规则

这些规则优先级高于所有下层文档：

1. 不要把 public leaderboard 反馈当成主要优化循环。
2. 没有明确的人类批准，不要自动提交 Kaggle。
3. 没有 OOF/CV 证据，不要声称有改进。
4. 生成 OOF 前，不要在 full train 上训练目标相关变换。
5. Target encoding 必须 OOF-safe，或在 fold 内做 nested 编码。
6. 每个 ensemble candidate 都必须有对齐的 OOF 预测。
7. 不要覆盖旧实验 artifact；创建新的 `experiment_id`。
8. 失败和被拒绝的实验也必须记录。
9. 不要把凭证、Kaggle token、原始数据和生成 artifact 提交到 git。
10. LLM 输出在经过代码、OOF、日志和 artifact 验证前，只能视为假设。

## 0x04. 进入仓库后的读取顺序

进入仓库后按顺序阅读：

1. `AGENTS.md`
2. `README.md`，如果存在
3. `configs/`
4. `competitions/<competition>/goal.yaml`，如果存在
5. `docs/` 下与当前任务相关的文档
6. 最近的 experiment registry 或 artifact manifest，如果存在

不要默认阅读所有深层文档。根据任务选择。

## 0x05. 任务路由

行动前先查这张表：

| 任务类型 | 下一步阅读 | 主要输出 |
| --- | --- | --- |
| 新 `/goal` 或验收标准 | `docs/goal_contract.md` | 解析后的 goal 与验收记录 |
| Agent 工作流 | `docs/agent_protocol.md` | 结构化任务图 |
| P0 项目基座 | `docs/p00_project_foundation.md` | 项目边界、规则与验收标准 |
| P1 实验证据流水线 | `docs/p01_evidence_pipeline.md` | OOF、submission、registry 证据链 |
| P2 配置实验工作台 | `docs/p02_experiment_workbench.md` | config-driven run 与 compare |
| P3 候选池与 ensemble | `docs/p03_candidate_ensemble.md` | OOF-backed candidate pool 和融合方案 |
| KGMON/S6E3 方法论 | `docs/kgmon_methodology.md` | proposal 或 plan |
| 特征工程 | `docs/feature_catalog.md` 和 `docs/leakage_rules.md` | feature proposal 或 config |
| 模型族选择 | `docs/model_zoo.md` | model config 或实验计划 |
| Ensemble 或 stacking | `docs/stacking.md` 和 `docs/artifact_contract.md` | OOF-backed ensemble plan |
| 泄漏、LB、提交安全 | `docs/leakage_rules.md` | 风险审查 |
| 路径、输出、manifest | `docs/artifact_contract.md` | artifact contract 或 registry 变更 |
| 依赖或运行环境 | `docs/environment_notes.md` | 环境建议 |
| 测试策略或测试数据集 | `docs/testing_strategy.md` | 测试计划或测试验收 |

## 0x06. 仓库地图

目标基线结构：

```text
kaggle-tabular-forge/
  AGENTS.md
  README.md
  pyproject.toml
  .gitignore
  .env.example
  configs/
    templates/
    registries/
    schemas/
  competitions/
    <competition>/
      goal.yaml
      configs/
      notebooks/
      reports/
      proposals/
  data/
    <competition>/
      raw/
      external/
      interim/
      processed/
      features/
  artifacts/
    experiments/
    oof/
    submissions/
    reports/
    llm_proposals/
    eda_findings/
    candidate_selection/
  src/
    ktabforge/
  scripts/
  tests/
  docs/
```

当前仓库不一定已经包含所有目录。随着契约变得可执行，再逐步添加结构。

## 0x07. 核心工作流摘要

默认 `/goal` 流程：

```text
/goal
  -> Director: 解析 objective 和约束
  -> Data Auditor: EDA 与泄漏扫描
  -> CV Architect: 验证协议
  -> Model Builder: baseline 或实验
  -> Experiment Analyst: 对比结果
  -> Feature Forge: 提出下一批特征
  -> Code Reviewer: 泄漏与可复现审查
  -> Infra and Cost: run mode 与预算
  -> Submission Strategist: 只有具备 OOF 证据时才准备候选提交
  -> Human: 批准任何 Kaggle submission
```

默认执行阶梯：

1. Static review。
2. Smoke test。
3. Single fold。
4. Full CV。
5. Repeated seeds、HPO、GPU scale-out 或 submission 只有在批准后执行。

## 0x08. 证据语言

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
- CV/LB gap needs investigation

## 0x09. 深层参考

需要时再读：

- `docs/goal_contract.md`: `/goal` 输入、自动发现规则、验收 schema、inconclusive 条件。
- `docs/README.md`: 文档索引；按阶段、契约、方法论做渐进式披露导航。
- `docs/p00_project_foundation.md`: P0 项目基座、规则、验收标准、非目标。
- `docs/p01_evidence_pipeline.md`: P1 实验证据流水线、smoke run、真实数据证据。
- `docs/p02_experiment_workbench.md`: P2 配置驱动实验工作台、model adapter、compare。
- `docs/p03_candidate_ensemble.md`: P3 候选池与轻量 ensemble 方案。
- `docs/agent_protocol.md`: AI Council 角色、Agent I/O、`/goal`、失败实验记录。
- `docs/kgmon_methodology.md`: S6E3 启发的方法论、MVP/增强范围、候选池思路。
- `docs/feature_catalog.md`: 特征家族与 feature registry 契约。
- `docs/model_zoo.md`: 模型家族、可选依赖理念、model registry 指南。
- `docs/stacking.md`: OOF candidates、hill climbing、logistic stacking、多层集成。
- `docs/leakage_rules.md`: target encoding、external data、public LB、submission 规则。
- `docs/artifact_contract.md`: artifact 路径、manifest、预测文件、experiment id。
- `docs/environment_notes.md`: 配置文件类型、依赖分组、Windows/WSL/Kaggle 边界。
- `docs/testing_strategy.md`: 测试层级、客户流失预测测试数据集、P0 测试验收。

## Prime Directive

大胆思考，谨慎实现，本地验证，诚实记录。

任何没有 CV/OOF、日志和可复现命令支撑的内容，都只是一个假设，不是结果。
