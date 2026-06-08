# Feature Catalog

Feature engineering is the primary driver of performance in this project. The
S6E3 solution shows that nearly all final models can share a common feature core
while specialized experiments explore extra feature families.

Every feature family should be registered with:

```yaml
feature_family: string
source_columns:
  - string
requires_external_data: boolean
requires_target: boolean
fold_safety: global_safe | fold_safe | high_risk
leakage_risk: low | medium | high
transductive_risk: low | medium | high
intended_models:
  - xgboost
  - lightgbm
  - catboost
  - neural_net
artifacts:
  manifest: path
  cache: path
validation_plan: string
```

## Snap Features

Map synthetic continuous values to nearest values in an original or external
reference dataset.

Example:

```python
MC_snap = nearest MonthlyCharges in original IBM data
TC_snap = nearest TotalCharges in original IBM data
MC_snap_diff = MonthlyCharges - MC_snap
TC_snap_diff = TotalCharges - TC_snap
```

Generalized form:

- `snap_<col>`: nearest reference value.
- `snap_diff_<col>`: synthetic residual or generator noise.
- `snap_rank_<col>`: rank of nearest reference match.
- `snap_distance_<col>`: distance to nearest reference value.

Purpose:

- recover latent original values
- measure generator perturbation
- create anchor keys for target encoding and nearest-neighbor lookup

## Digit and Decimal Features

Extract hidden signal from numeric string and decimal artifacts.

Examples:

```python
frac = x - floor(x)
d1 = floor(frac * 10)
d2 = floor(frac * 100) % 10
frac100 = round(frac * 100)
mod10 = floor(x) % 10
mod100 = floor(x) % 100
```

Variants:

- decimal digits `d1` to `dN`
- fractional part
- integer modulo features
- common-denominator residuals: `1/2`, `1/4`, `1/5`, `1/10`
- rounded flags such as `frac < 0.005`
- digit-pair categorical strings
- character n-grams from numeric strings

Purpose:

- expose synthetic generator fingerprints
- help tree models split on hidden decimal patterns
- give neural models explicit numeric artifacts they may not discover alone

## Nested Target Encoding

Target encoding must be nested and leak-free.

Use cases:

- raw categorical columns
- binned numeric columns
- bigram and trigram categorical crosses
- anchor keys such as `(snap_value, tenure)`
- numeric by categorical snap products

Statistics beyond mean:

- standard deviation
- minimum and maximum
- median
- quantiles: 5th, 10th, 45th, 55th, 90th, 95th

Rules:

- For OOF predictions, fit target encoders inside each training fold only.
- When using nested TE, use an inner fold loop inside each outer CV fold.
- External original-data priors are allowed only when source and leakage status
  are documented.
- CatBoost native ordered target statistics can replace manual TE for raw
  categorical columns, but derived TE features still need explicit review.

## Arithmetic Interaction Features

Capture consistency constraints and domain formulas.

Example:

```python
TC_deviation = TotalCharges - tenure * MonthlyCharges
TC_snap_exp_dev = TC_snap - tenure * MC_snap
TC_per_month = TotalCharges / (tenure + 1)
MC_to_TC_ratio = MonthlyCharges / (TotalCharges + 1e-9)
```

Generalized form:

- actual vs expected totals
- ratios
- residuals
- per-unit quantities
- billing or time-consistency anomalies

Purpose:

- expose domain equation violations
- measure synthetic perturbation
- create compact high-signal numeric features

## Multi-Scale Binning

Convert continuous columns into categorical bins at several resolutions.

Variants:

- quantile bins
- high-resolution quantile bins
- fixed-width bins
- log-scale bins
- integer floor bins
- snap-value bins

Rules:

- Bins used with target statistics must follow fold-safe encoding.
- Very high bin counts should be cost-labeled and guarded against sparse noise.
- Binned features should be recorded in the feature manifest with binning
  parameters.

## Categorical Cross Features

Create joint categorical profiles that single-column encodings cannot capture.

Examples:

```python
bi_Contract_Internet = Contract + "__" + InternetService
tri_Contract_Internet_Payment = Contract + "__" + InternetService + "__" + PaymentMethod
```

Variants:

- all selected pairs
- selected triples
- high-signal domain crosses
- numeric-bin by categorical crosses

Rules:

- Crosses should be selected by EDA, domain logic, cardinality, or prior
  experiment evidence.
- All target-dependent encodings of crosses must be nested or OOF-safe.
- Record cardinality and rare-category handling.

## Frequency and Count Encoding

Measure how often each value appears in train, test, original data, or combined
unsupervised data.

Examples:

```python
freq = all_data[col].value_counts(normalize=True)
all_data[f"freq_{col}"] = all_data[col].map(freq)
```

Variants:

- normalized frequency
- raw count
- train/test count ratio
- synthetic/original count ratio
- cluster density around snap values
- drift ratio: `log1p(train_freq / orig_freq)`

Risk:

- Train+test frequency encoding is transductive. It can be useful but must be
  labeled.

## Service Count and Binary Aggregations

Aggregate related binary or categorical service columns.

Example:

```python
svc_yes = sum((df[c] == "Yes").astype(int) for c in svc_cols)
```

Variants:

- number of enabled services
- number of disabled services
- has internet
- has phone
- per-service `ISYES_`, `ISNO_`, `ISOTHER_` flags
- grouped missing counts
- option-count features
- selected domain aggregates

## Original or External Dataset Lookup

Use legal external/original data as an anchor.

S6E3 pattern:

- Build a nearest-neighbor index on original rows.
- Find nearest original row for each synthetic row.
- Attach allowed original labels, priors, distances, or identifiers as features.

Generalized forms:

- `external_anchor_lookup`
- `nearest_original_neighbor_features`
- `original_prior_features`
- `prototype_distance_features`
- `reference_manifold_features`

Rules:

- External data source, license, and relationship to competition data must be
  documented.
- Label use from external data must receive leakage review.
- Nearest-neighbor features must record columns, scaling, distance metric, and
  index type.

## Radix Interaction Features

Encode continuous by categorical pairs into single integer categories.

Example:

```python
radix = int(MC_snap * 100) + cat_code * 100_000
```

Purpose:

- provide tree models a compact split-friendly representation
- represent numeric and categorical interactions explicitly
- avoid requiring chained splits to discover the pair

Rules:

- The radix base must prevent collisions.
- The manifest must record scaling and category code mapping.

## Synthetic Artifact Detection

Detect generator fingerprints that do not occur naturally.

Feature families:

- `intlike_count`: count values with `frac < 0.001`
- `quarterlike_count`
- `halflike_count`
- rounding flags
- repeated decimal pattern features
- numeric string TF-IDF character n-grams
- Benford-like leading-digit deviation
- train/original drift ratios

Purpose:

- identify rows or values shaped by the generator
- expose hidden sampling rules
- create model diversity through artifact-aware pipelines

## Projection and Manifold Features

Represent how each row relates to original or reference data geometry.

Variants:

- PCA on original/reference data
- Gaussian random projection
- KMeans prototype distances
- KNN distances and neighbor statistics
- DAE reconstruction error
- DAE latent embeddings
- cyclical domain features such as `sin/cos(tenure * 2pi / period)`

Rules:

- If fitted on target-independent data, label as unsupervised.
- If fitted using train+test, label transductive risk.
- If fitted on original data, document source and columns.
