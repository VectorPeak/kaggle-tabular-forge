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

注意：示例里的 `experiment_id` 是固定的。重复运行同一个配置会因为 artifact 不允许覆盖而失败；需要重新运行时，请复制配置并改一个新的 `experiment_id`。
