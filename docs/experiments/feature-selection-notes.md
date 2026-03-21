# Feature Selection Notes

Feature selection review points:

- Remove constant and near-constant columns early.
- Inspect missingness patterns before imputation.
- Compare feature stability across folds.
- Treat high-cardinality categorical features carefully.
- Keep rejected features documented when the reason is leakage or instability.

The goal is to make feature pruning reproducible rather than dependent on memory.
