# March Experiment Workflow Notes

A stable tabular experiment should keep artifacts reproducible from the first baseline.

1. Build a small EDA summary before feature work.
2. Define the validation split and leakage rules before model selection.
3. Track out-of-fold predictions for every candidate model.
4. Save feature lists together with each experiment result.
5. Compare submissions only after local evidence is recorded.

This workflow is intentionally slower than ad-hoc modeling, but it keeps later stacking and review work reliable.
