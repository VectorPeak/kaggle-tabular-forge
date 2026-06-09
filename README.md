# kaggle-tabular-forge

`kaggle-tabular-forge` 是一个面向 Kaggle 表格赛的可复用实验框架。它的核心不是黑箱 AutoML，而是：用严格的 CV/OOF 证据、可追踪 artifact、实验 registry 和安全 gate，把特征、模型、ensemble 想法一步步锻造成可信结果。

项目受到 NVIDIA/KGMON S6E3 这类高吞吐竞赛系统启发：人类制定策略，LLM 辅助提出假设，所有结论都必须经过 OOF 和 artifact 证明。

## 当前阶段

项目已经从 P1 最小可运行阶段进入 P2 实验工作台阶段，并完成 P3 候选池与轻量 ensemble 的 SRC MVP。

- P0：项目基座、规则和验收标准，回答“Agent 与实验边界是否清楚？”
- P1：实验证据流水线，回答“能否跑出可审计的 OOF、submission 和 registry？”
- P2：配置驱动实验工作台，回答“能否批量运行、比较和追踪多个实验？”
- P3：候选池与轻量 ensemble，回答“能否从可信候选中安全构造融合结果？”当前 MVP 已实现。
- P4：候选实验工厂，回答“能否从 matrix 配置批量生产可信 P2 候选？”当前 MVP 已实现。

当前 P2 已经具备：

- `uv`/`pyproject.toml` 项目环境。
- `src/ktabforge` Python 包。
- `ktab` CLI。
- 配置 schema 校验。
- `ktab run --config` 配置驱动实验入口。
- `ktab compare` registry 对比入口。
- tiny churn fixture。
- StratifiedKFold OOF baseline。
- Logistic Regression smoke baseline。
- LightGBM smoke baseline。
- OOF、submission、metrics、manifest、registry artifact 写入。
- config snapshot 与 config hash。
- pytest 与 ruff 验证。

## 快速开始

```powershell
cd E:\Github\kaggle-tabular-forge
uv sync --group dev
uv run pytest
uv run ruff check .
```

查看 CLI：

```powershell
uv run ktab --help
```

校验配置：

```powershell
uv run ktab validate-config `
  --config configs\competition.example.yaml `
  --schema configs\schemas\competition.schema.json
```

运行 P1 tiny smoke：

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

运行 P2 配置驱动实验：

```powershell
uv run ktab run --config configs\experiments\p02_churn_logistic_basic.example.yaml
```

运行 LightGBM 示例前先安装可选依赖：

```powershell
uv sync --group dev --extra lgbm
uv run ktab run --config configs\experiments\p02_churn_lgbm_basic.example.yaml
```

比较已有实验：

```powershell
uv run ktab compare --artifact-root artifacts --competition playground-series-s6e3 --top-n 10
```

运行 P3 ensemble：

```powershell
uv run ktab ensemble --config configs\ensembles\p03_candidate_ensemble.example.yaml
```

运行 P4 candidate factory dry-run：

```powershell
uv run ktab factory --config configs\matrices\p04_churn_candidate_factory.example.yaml --dry-run
```

生成的本地 artifact 会写入：

```text
artifacts/
├── experiments/<competition>/<experiment_id>/      # metrics、manifest、fold metrics、review
├── oof/<competition>/<experiment_id>/              # oof.parquet
├── submissions/<competition>/<experiment_id>/      # submission.csv
└── registry/<competition>/                         # experiment_registry.csv
```

`artifacts/` 和真实 `data/` 默认不提交到 git。

## `/goal` 约定

人类输入应保持简短：

```text
/goal objective=<objective> competition=<competition> request="<intent>" budget=<budget>
```

示例：

```text
/goal objective=build_baseline competition=playground-series-s6e3 request="build first honest LightGBM baseline with OOF artifacts" budget=single_fold
```

系统应先从 configs、registry 和已有 artifact 中自动发现 metric、target、路径、CV 默认值、当前最佳实验和 artifact 路径，然后再追问缺失信息。

完整契约见 [docs/goal_contract.md](docs/goal_contract.md)。

## 测试数据

默认测试使用仓库内 tiny churn fixture，不依赖真实 Kaggle 下载：

```text
tests/fixtures/data/churn_tiny/
├── train.csv
├── test.csv
└── sample_submission.csv
```

真实 Kaggle 客户流失数据后续作为 `local_data` 层接入，推荐本地路径：

```text
data/playground-series-s6e3/raw/train.csv
data/playground-series-s6e3/raw/test.csv
data/playground-series-s6e3/raw/sample_submission.csv
```

## 证据规则

一个实验只有具备对应证据时才算可信：

- config
- command
- CV protocol
- OOF predictions
- fold metrics
- test predictions 或 submission
- feature/model manifests
- leakage review
- reproducibility metadata
- experiment registry record

没有 OOF/CV 证据、日志和可复现命令时，任何结论都只能算 hypothesis。

## 文档索引

入口文档：

- [AGENTS.md](AGENTS.md)：Agent 不可违反的规则与任务路由。
- [README.md](README.md)：人类入口、项目阶段、快速命令。

核心契约：

- [docs/README.md](docs/README.md)：文档索引与渐进式披露导航。
- [docs/goal_contract.md](docs/goal_contract.md)：`/goal` 输入与验收。
- [docs/p00_project_foundation.md](docs/p00_project_foundation.md)：P0 项目基座、规则与验收标准。
- [docs/p01_evidence_pipeline.md](docs/p01_evidence_pipeline.md)：P1 最小可运行证据流水线。
- [docs/p02_experiment_workbench.md](docs/p02_experiment_workbench.md)：P2 配置驱动实验工作台。
- [docs/p03_candidate_ensemble.md](docs/p03_candidate_ensemble.md)：P3 候选池与轻量 ensemble 方案。
- [docs/p04_candidate_factory.md](docs/p04_candidate_factory.md)：P4 候选实验工厂。
- [docs/agent_integration.md](docs/agent_integration.md)：Codex 多 Agent / Skill / MCP 集成想法。
- [docs/artifact_contract.md](docs/artifact_contract.md)：artifact 布局与 OOF 对齐。
- [docs/leakage_rules.md](docs/leakage_rules.md)：泄漏与 public leaderboard 规则。
- [docs/testing_strategy.md](docs/testing_strategy.md)：测试层级与客户流失数据集。

方法论与系统设计：

- [docs/kgmon_methodology.md](docs/kgmon_methodology.md)：方法论与阶段路线图。
- [docs/agent_protocol.md](docs/agent_protocol.md)：Agent 角色、输入输出与任务图。
- [docs/feature_catalog.md](docs/feature_catalog.md)：特征家族与风险标签。
- [docs/model_zoo.md](docs/model_zoo.md)：模型家族与依赖分组。
- [docs/stacking.md](docs/stacking.md)：OOF-backed ensemble、hill climbing、stacking。
- [docs/environment_notes.md](docs/environment_notes.md)：环境、依赖、Windows/WSL/Kaggle 边界。

## 当前项目结构

```text
kaggle-tabular-forge/
├── AGENTS.md                         # Agent 总纲
├── README.md                         # 人类入口
├── pyproject.toml                    # uv/Python 项目配置
├── configs/                          # 人类可编辑配置与 JSON Schema
├── docs/                             # 设计、契约、路线图
├── src/ktabforge/                    # P1 Python 包
└── tests/                            # unit/integration/smoke 测试
```
