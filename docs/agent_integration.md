# Agent Integration

本文档记录 `kaggle-tabular-forge` 和 Codex 多 Agent / 插件 / MCP 工具结合的产品想法。

## Core Principle

最终形态不是让 Agent 随意写 notebook 或自由训练模型，而是分层协作：

```text
Agent 负责想、审、拆、记录
ktab 负责跑、验、产出证据
```

Agent 可以提出实验假设、生成配置、审查泄漏、总结报告；真正执行实验、保存 OOF、写 registry、生成 submission 的动作必须通过 `ktab` CLI 或结构化工具完成。

## Product Shape

推荐最终形态：

```text
kaggle-tabular-forge
├── Python package             # 核心实验执行库
├── CLI: ktab                  # 人类和 Agent 共用的命令入口
├── GitHub template repo       # 新比赛复用的仓库模板
├── Codex Skill / Plugin       # 把流程嵌入 Codex 多 Agent
└── MCP server / tools         # 给 Agent 暴露结构化工具接口
```

## Agent Roles

P3/P4 之后可以引入以下 Agent 分工：

```text
EDA Agent
# 读取数据摘要，提出可疑关系、分布差异和特征方向

Feature Agent
# 提出特征家族，但必须标注泄漏风险和 fold-safety

Model Agent
# 提出模型、preset、参数和预算

Runner Agent
# 只负责调用 ktab run，不直接绕过 evidence pipeline

Review Agent
# 检查 OOF、artifact、leakage、registry 和结论可信度

Report Agent
# 总结本轮实验，写入 docs、README 或学习日志
```

## Tool Interface

未来可以把 `ktab` 能力暴露为 MCP tools：

```text
validate_config(config_path)
run_experiment(config_path)
compare_experiments(competition, top_n)
list_artifacts(competition)
read_metrics(experiment_id)
read_registry(competition)
create_experiment_config(model, features, seed)
review_leakage(experiment_id)
generate_report(experiment_id)
```

这样 Agent 拿到的是结构化结果，而不是靠猜 shell 输出。

## Codex Skill Pack

可以把项目打包成一个 Codex Skill：

```text
kaggle-tabular-forge-vp
```

触发场景：

```text
跑一个表格赛 baseline
审核实验结果
给我下一批 feature/model 方案
比较这些 OOF
记录这次 Kaggle 经验
生成下一轮 /goal
```

Skill 工作流：

```text
1. 读取 AGENTS.md
2. 读取 README.md
3. 按 AGENTS 路由或 docs/README.md 进入相关深层文档
4. 校验 configs
5. 调用 ktab run
6. 调用 ktab compare
7. 审查 artifact 和 leakage
8. 写实验报告
9. 更新学习日志
```

## Safety Rules

Agent 集成必须遵守：

- 不直接根据 public leaderboard 自动调参。
- 不自动提交 Kaggle。
- 没有 OOF 的实验不能 claim improvement。
- 有 target-dependent 特征时必须通过 leakage review。
- Agent 只能提案或调用工具，不能绕过 evidence pipeline。
- 所有实验都必须落到 artifact + registry。

## Roadmap

建议路线：

```text
P2
# CLI + config-driven run + compare，当前已开始落地

P3
# feature/model 扩展、更多候选实验、基础 ensemble

P4
# Codex Skill Pack，多 Agent 提案与审查

P5
# MCP server，把 ktab 能力暴露为结构化工具

P6
# GitHub template + packaged release，形成可复用产品
```

## Product Positioning

一句话定位：

> 一个面向 Kaggle 表格赛的 Agent-native 实验工作台。

它不是普通 AutoML，也不是普通模板仓库，而是：

```text
人类制定目标
Agent 提出假设
ktab 执行实验
OOF 证明结果
registry 记录历史
Agent 总结下一轮
```
