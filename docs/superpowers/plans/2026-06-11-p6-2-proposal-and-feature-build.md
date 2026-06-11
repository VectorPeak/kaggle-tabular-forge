# P6.2 Proposal and Feature Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `kaggle-tabular-forge` 增加 P6.2 的最小闭环：`proposal validate/register` 和 `feature build`，支持 `frequency/count`、`binning`、`arithmetic_interactions` 三类低风险特征。

**Architecture:** 保持仓库现有“薄 CLI + typed config + runner 落盘”的模式。`proposal` 作为结构化提案层，负责 schema 校验与 artifact 注册；`feature build` 作为独立执行层，读取原始标准表格数据，构建增强版 train/test 特征表，并写出 manifest、schema、report。第一版不强行改写 P2 runner 消费 feature bundle，只把 P6.2 的构建闭环打稳。

**Tech Stack:** Python 3.12, Typer, pandas, jsonschema, PyYAML, pytest, ruff

---

## 文件结构

### 新增文件

- `src/ktabforge/proposals/__init__.py`
- `src/ktabforge/proposals/config.py`
- `src/ktabforge/proposals/runner.py`
- `src/ktabforge/features/pipeline.py`
- `src/ktabforge/features/registry.py`
- `src/ktabforge/features/transforms/__init__.py`
- `src/ktabforge/features/transforms/base.py`
- `src/ktabforge/features/transforms/frequency.py`
- `src/ktabforge/features/transforms/binning.py`
- `src/ktabforge/features/transforms/interactions.py`
- `configs/schemas/proposal.schema.json`
- `configs/schemas/feature_pipeline.schema.json`
- `configs/proposals/p06_freq_count.example.yaml`
- `configs/features/p06_freq_bundle.example.yaml`
- `tests/unit/test_proposal_config.py`
- `tests/unit/test_feature_build_config.py`
- `tests/unit/test_feature_transforms.py`
- `tests/integration/test_p6_proposal_register.py`
- `tests/integration/test_p6_feature_build.py`

### 修改文件

- `src/ktabforge/cli.py`
- `tests/integration/test_cli.py`
- `tests/unit/test_config_schema.py`
- `configs/README.md`

## 任务 1：Proposal 配置、CLI 与注册 artifact

**Files:**
- Create: `src/ktabforge/proposals/__init__.py`
- Create: `src/ktabforge/proposals/config.py`
- Create: `src/ktabforge/proposals/runner.py`
- Create: `configs/schemas/proposal.schema.json`
- Create: `configs/proposals/p06_freq_count.example.yaml`
- Create: `tests/unit/test_proposal_config.py`
- Create: `tests/integration/test_p6_proposal_register.py`
- Modify: `src/ktabforge/cli.py`
- Modify: `tests/integration/test_cli.py`
- Modify: `tests/unit/test_config_schema.py`

- [ ] 先写 proposal config 与 register 的失败测试，覆盖：
  - example config 通过 schema
  - `proposal_id` / `competition` / `feature_family` / `source_columns` / `hypothesis` 必填
  - `ktab proposal validate --config ...` 可返回 `valid`
  - `ktab proposal register --config ...` 写出 manifest 与 config snapshot

- [ ] 运行失败测试：

```powershell
uv run pytest tests/unit/test_proposal_config.py tests/integration/test_p6_proposal_register.py tests/integration/test_cli.py tests/unit/test_config_schema.py -q
```

- [ ] 实现 `ProposalConfig`、schema、example config，以及 CLI：
  - `ktab proposal validate --config <proposal.yaml>`
  - `ktab proposal register --config <proposal.yaml>`

- [ ] `proposal register` 写出：

```text
artifacts/proposals/<competition>/<proposal_id>/
  proposal_manifest.json
  proposal_config.yaml
```

- [ ] 复跑 proposal 相关测试并确认通过。

## 任务 2：Feature build 配置、变换注册与构建 runner

**Files:**
- Create: `src/ktabforge/features/pipeline.py`
- Create: `src/ktabforge/features/registry.py`
- Create: `src/ktabforge/features/transforms/__init__.py`
- Create: `src/ktabforge/features/transforms/base.py`
- Create: `src/ktabforge/features/transforms/frequency.py`
- Create: `src/ktabforge/features/transforms/binning.py`
- Create: `src/ktabforge/features/transforms/interactions.py`
- Create: `configs/schemas/feature_pipeline.schema.json`
- Create: `configs/features/p06_freq_bundle.example.yaml`
- Create: `tests/unit/test_feature_build_config.py`
- Create: `tests/unit/test_feature_transforms.py`
- Create: `tests/integration/test_p6_feature_build.py`
- Modify: `src/ktabforge/cli.py`
- Modify: `tests/unit/test_config_schema.py`

- [ ] 先写 feature build 的失败测试，覆盖：
  - example config 通过 schema
  - `ktab feature build --config ...` 写出 artifact
  - `frequency/count`、`binning`、`arithmetic_interactions` 三类变换都能生成预期列
  - train/test 行数保持一致，`id` 保持不变，train 保留 target

- [ ] 运行失败测试：

```powershell
uv run pytest tests/unit/test_feature_build_config.py tests/unit/test_feature_transforms.py tests/integration/test_p6_feature_build.py tests/unit/test_config_schema.py -q
```

- [ ] 实现最小 feature build config 与 pipeline：
  - 输入为标准 `train.csv` / `test.csv` / `sample_submission.csv`
  - 输出为增强版 `train_features.parquet` / `test_features.parquet`
  - 三类变换支持显式 step 配置

- [ ] `feature build` 写出：

```text
artifacts/features/<competition>/<feature_build_id>/
  train_features.parquet
  test_features.parquet
  feature_build_manifest.json
  feature_schema.json
  feature_build_report.md
```

- [ ] 复跑 feature build 相关测试并确认通过。

## 任务 3：文档与阶段回归

**Files:**
- Modify: `configs/README.md`

- [ ] 在 `configs/README.md` 中补 proposal 与 feature build 示例命令。

- [ ] 跑 P6 相关回归：

```powershell
uv run pytest tests/unit/test_eda_config.py tests/unit/test_eda_profiling.py tests/unit/test_eda_watchlist.py tests/unit/test_proposal_config.py tests/unit/test_feature_build_config.py tests/unit/test_feature_transforms.py tests/integration/test_p6_eda_scan.py tests/integration/test_p6_proposal_register.py tests/integration/test_p6_feature_build.py tests/integration/test_cli.py tests/unit/test_config_schema.py -q
```

## 最终回归

- [ ] 运行完整测试：

```powershell
uv run pytest -q
```

- [ ] 运行 lint：

```powershell
uv run ruff check .
```

- [ ] 提交并推送：

```powershell
git add src/ktabforge/proposals src/ktabforge/features src/ktabforge/cli.py configs/proposals configs/features configs/schemas/proposal.schema.json configs/schemas/feature_pipeline.schema.json tests configs/README.md docs/superpowers/plans/2026-06-11-p6-2-proposal-and-feature-build.md
git commit -m "Implement P6.2 proposal and feature build"
git push
```
