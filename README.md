# kaggle-tabular-forge

`kaggle-tabular-forge` 是一个面向 Kaggle 表格赛的可复用实验框架。它的目标不是做一个“黑盒 AutoML”，而是把表格赛里最容易失真的部分先规范起来：配置、交叉验证、OOF 证据、artifact、实验对比、候选池和 ensemble。

这个仓库当前已经完成 P0-P5 的基础骨架，重点是把“能跑”变成“可审计、可复现、可扩展”。

## 项目目标

我们希望把一场表格赛拆成几层稳定能力：

- 证据优先：任何结论都要有 OOF、CV 和 artifact 支撑
- 配置驱动：实验不靠散落脚本，尽量通过 YAML 配置描述
- 候选池化：先批量产生候选，再做筛选、融合和堆叠
- 对 Agent 友好：便于 Codex / LLM Agent 在明确边界内协作
- 对新比赛可迁移：换数据集时，尽量复用同一套工作流

## 当前阶段

目前仓库已经覆盖以下阶段：

- `P0` 项目基座：文档、规则、验收标准、配置约定
- `P1` 实验证据流水线：最小可运行的 OOF / submission / registry 流程
- `P2` 配置驱动实验工作台：通过 YAML 运行、比较和追踪实验
- `P3` 候选池与轻量 ensemble：基于 OOF 的安全融合能力
- `P4` 候选实验工厂：通过 matrix 配置批量生成候选实验
- `P5` stacking / selection 基础能力：配置约束、候选筛选、hill climbing、OOF-safe 一层 stacking

当前更适合把下一阶段理解为：

- `P6`：更重的 EDA 驱动特征工程工作台
- `P7`：pseudo-label、extra training 和 final candidate workflow

## 技术原则

这个项目有几个硬约束：

- 没有 OOF / CV 证据的提升，不算有效提升
- 不允许把 leaderboard 偶然性当成真实收益
- 实验结果必须能回溯到配置、命令、模型、特征和输出文件
- 先做最小但诚实的 baseline，再逐步扩展复杂度
- 任何自动化能力都要服从泄漏检查和可复现边界

## 快速开始

推荐使用 `uv` 管理 Python 环境。

```powershell
cd E:\Github\kaggle-tabular-forge
uv sync --group dev
uv run pytest -q
uv run ruff check .
```

查看 CLI：

```powershell
uv run ktab --help
```

## 常用命令

### 1. 校验配置

```powershell
uv run ktab validate-config `
  --config configs\experiment.example.yaml `
  --schema configs\schemas\experiment.schema.json
```

### 2. 运行 P1 smoke 流水线

这一步使用仓库自带的 tiny churn fixture，不依赖外部 Kaggle 下载。

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

### 3. 运行 P2 配置驱动实验

```powershell
uv run ktab run --config configs\experiments\p02_churn_logistic_basic.example.yaml
```

如果要运行 LightGBM 示例，需要额外安装可选依赖：

```powershell
uv sync --group dev --extra lgbm
uv run ktab run --config configs\experiments\p02_churn_lgbm_basic.example.yaml
```

### 4. 比较实验结果

```powershell
uv run ktab compare --artifact-root artifacts --competition playground-series-s6e3 --top-n 10
```

### 5. 运行 P3 ensemble

```powershell
uv run ktab ensemble --config configs\ensembles\p03_candidate_ensemble.example.yaml
```

### 6. 运行 P4 candidate factory

先 dry-run 看计划：

```powershell
uv run ktab factory `
  --config configs\matrices\p04_churn_candidate_factory.example.yaml `
  --dry-run
```

正式执行时去掉 `--dry-run`，也可以加 `--max-runs` 限制本次运行数量。

### 7. 运行 P5 stacking preflight

```powershell
uv run ktab stack-preflight --config configs\stacking.example.yaml
```

### 8. 运行 P5 completed stacking

```powershell
uv run ktab stack --config configs\stacking.example.yaml
```

更完整的阶段说明见：

- [docs/p05_stacking_foundation.md](docs/p05_stacking_foundation.md)
- [docs/p06_feature_workbench.md](docs/p06_feature_workbench.md)

## 依赖分组

基础依赖已经包含：

- `numpy`
- `pandas`
- `pyarrow`
- `scikit-learn`
- `typer`
- `jsonschema`
- `pyyaml`
- `rich`
- `joblib`

可选依赖通过 `pyproject.toml` 分组管理：

- `lgbm`：LightGBM
- `xgb`：XGBoost
- `cat`：CatBoost
- `gbdt`：常见 GBDT 家族合集
- `llm`：`openai` / `anthropic` 等 Agent 能力
- `kaggle`：Kaggle API 相关依赖
- `viz`：可视化依赖

示例：

```powershell
uv sync --group dev --extra lgbm --extra llm
```

## 数据与 Artifact

本仓库默认区分两类数据：

- `tests/fixtures/data/`：仓库内测试数据，用于 smoke / unit / integration
- `data/<competition>/raw/`：你本地放置的真实比赛数据，不提交到 git

推荐的真实数据目录：

```text
data/
└── playground-series-s6e3/
    └── raw/
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv
```

运行产物会写入 `artifacts/`，大致结构如下：

```text
artifacts/
├── experiments/<competition>/<experiment_id>/    # metrics, manifest, fold metrics, review
├── oof/<competition>/<experiment_id>/            # oof.parquet
├── submissions/<competition>/<experiment_id>/    # submission.csv
├── registry/<competition>/                       # experiment_registry.csv
└── failures/<competition>/<experiment_id>/       # 失败运行记录
```

## 项目结构

```text
kaggle-tabular-forge/
├── AGENTS.md                 # Agent 总纲与最高优先级规则
├── README.md                 # 人类主入口
├── pyproject.toml            # uv / Python 项目配置
├── configs/                  # YAML 配置与 JSON Schema
├── docs/                     # 方案、契约、路线图与方法论
├── src/ktabforge/            # 核心源码
├── tests/                    # 单测、集成测试、fixture
├── data/                     # 本地真实比赛数据
└── artifacts/                # 实验产物输出
```

`src/ktabforge/` 里当前主要模块可以这样理解：

- `config/`：配置加载、schema 校验、安全命名约束
- `pipeline/`：实验运行主流程
- `artifacts/`：输出文件布局、manifest、失败记录
- `models/`：模型注册与基线实现
- `features/`：特征集定义
- `ensembles/`：P3 融合逻辑
- `factory/`：P4 候选实验工厂
- `reports/`：结果比较与汇总
- `safety/`：泄漏和运行安全规则

## 文档导航

如果你要快速理解整个仓库，建议按下面顺序看：

1. [AGENTS.md](AGENTS.md)
2. [docs/README.md](docs/README.md)
3. [docs/p00_project_foundation.md](docs/p00_project_foundation.md)
4. [docs/p01_evidence_pipeline.md](docs/p01_evidence_pipeline.md)
5. [docs/p02_experiment_workbench.md](docs/p02_experiment_workbench.md)
6. [docs/p03_candidate_ensemble.md](docs/p03_candidate_ensemble.md)
7. [docs/p04_candidate_factory.md](docs/p04_candidate_factory.md)
8. [docs/p05_stacking_foundation.md](docs/p05_stacking_foundation.md)
9. [docs/p06_feature_workbench.md](docs/p06_feature_workbench.md)

几个高频契约文档：

- [docs/goal_contract.md](docs/goal_contract.md)
- [docs/artifact_contract.md](docs/artifact_contract.md)
- [docs/leakage_rules.md](docs/leakage_rules.md)
- [docs/testing_strategy.md](docs/testing_strategy.md)
- [docs/agent_protocol.md](docs/agent_protocol.md)

## `/goal` 约定

这个仓库预留了面向 Agent 的目标驱动入口，输入可以尽量短：

```text
/goal objective=<objective> competition=<competition> request="<intent>" budget=<budget>
```

示例：

```text
/goal objective=build_baseline competition=playground-series-s6e3 request="build first honest LightGBM baseline with OOF artifacts" budget=single_fold
```

完整约定见 [docs/goal_contract.md](docs/goal_contract.md)。

## 测试

当前建议至少执行这三类检查：

```powershell
uv run pytest -q
uv run ruff check .
uv run ktab validate-config --config configs\experiment.example.yaml --schema configs\schemas\experiment.schema.json
```

如果你改动了 ensemble 或 factory，建议额外跑对应的集成测试和示例配置。

## 适合谁用

这个仓库更适合下面几类场景：

- 想把 Kaggle 表格赛流程沉淀成长期可复用工程的人
- 想让 LLM / Agent 参与实验，但不想让流程变成黑箱的人
- 想先把证据链、配置、artifact 和对比系统搭好的团队
- 想从单场比赛脚本堆，演化到通用实验平台的人

## 当前状态说明

这不是一个已经“功能全部完成”的通用 AutoML 产品。更准确地说，它是一个正在成型的、以证据和流程为核心的 Kaggle 表格赛工作台。

如果你现在要开始用它，最合适的方式是：

1. 用 `tests/fixtures/data/churn_tiny` 跑通最小流程
2. 用 `configs/experiments/` 下的示例配置跑通 P2
3. 再接入你自己的比赛数据和候选实验矩阵

后续如果需要，我可以继续把 `docs/README.md`、`configs/README.md` 和各阶段文档一起统一整理成完全中文化版本。
