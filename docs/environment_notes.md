# Environment Notes

Keep the base install light. Heavy libraries should be optional.

## Configuration Files

Use YAML for human-edited project configuration and JSON Schema for machine
validation.

Rules:

- YAML files under `configs/` are the source of human intent.
- JSON Schema files under `configs/schemas/` define the structural contract.
- JSON should be used for machine-written artifacts such as metrics, manifests,
  environment records, and registry rows.
- Do not use JSON as the primary hand-edited experiment format unless a tool
  specifically requires it.
- Keep credentials and tokens out of all config files; use environment variables
  and `.env.example`.

Suggested config groups:

- `configs/templates/`: reusable human-edited templates.
- `configs/registries/`: machine-readable global registries.
- `configs/schemas/`: JSON Schema contracts.
- `competitions/<competition>/configs/`: competition-specific configs.

## Dependency Groups

Suggested optional dependency groups:

```toml
[project.optional-dependencies]
core = [
  "numpy",
  "pandas",
  "polars",
  "pyarrow",
  "scikit-learn",
  "scipy",
  "joblib",
  "pyyaml",
  "pydantic",
  "rich",
  "tqdm",
  "matplotlib",
  "seaborn",
]
gbdt = [
  "lightgbm",
  "xgboost",
  "catboost",
  "optuna",
]
gpu = [
  "cupy-cuda12x",
  "cudf-cu12",
  "cuml-cu12",
]
nn = [
  "torch",
  "pytorch-lightning",
  "torchmetrics",
  "tabulate",
]
llm = [
  "openai",
  "anthropic",
  "instructor",
  "tenacity",
  "jinja2",
  "jsonschema",
]
kaggle = [
  "kaggle",
  "kagglehub",
]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
  "mypy",
  "pre-commit",
  "ipykernel",
  "jupyterlab",
  "nbstripout",
]
```

## Runtime Boundaries

Windows:

- suitable for docs, CPU baselines, config validation, and light EDA
- avoid installing RAPIDS directly in the default Windows Python environment
- use `pathlib.Path`; do not hardcode path separators

WSL or Linux GPU:

- preferred for RAPIDS, cuDF, cuML, and GPU-heavy experiments
- keep GPU dependency installation separate from CPU/dev installs
- record CUDA, driver, package versions, and hardware metadata

Kaggle:

- treat `/kaggle/input` as read-only
- write generated artifacts to `/kaggle/working`
- mirror final artifacts back into the repository only as manifests or selected
  reports, not raw heavy files

## Git Hygiene

Do not commit:

- `.env`
- Kaggle tokens
- API keys
- raw datasets
- generated artifacts
- notebook checkpoints
- large model binaries
- local cache directories

Prefer repo-relative paths in configs unless the runtime profile explicitly
requires an absolute path.
