# Leakage Check Assumptions

Leakage checks should run before expensive model iterations.

- Inspect timestamp, ID, target-derived, and post-event columns.
- Compare train and test cardinality for categorical features.
- Check whether missingness itself encodes the target.
- Treat target encoding as split-dependent and never fit it globally.
- Keep a reject list for suspicious features with written reasons.

The goal is not to remove useful signal blindly, but to make validation evidence trustworthy.
