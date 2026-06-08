# Model Zoo

Models should be grouped by implementation maturity and intended role. Do not
install or run the entire model zoo by default.

## MVP Model Set

The MVP should support:

- Logistic regression or ridge-style baseline.
- XGBoost.
- LightGBM.
- CatBoost.
- Simple weighted or rank average ensemble.
- Simple OOF-backed logistic stacker.

## Tree Model Families

Tree models are the first major model block for tabular data.

### XGBoost

Useful roles:

- strong baseline candidate
- manual TE consumer
- digit and decimal feature consumer
- high-cardinality engineered feature consumer
- pairwise ranking objective for AUC-like metrics

S6E3-inspired variants:

- nested TE on categorical columns
- anchor-based TE using snap keys
- digit position features
- combined high-signal feature unions
- bigram/trigram TE
- GPU pair target encoding via cuML
- multi-seed rank blending
- `rank:pairwise` plus calibration for AUC optimization

### LightGBM

Useful roles:

- fast baseline
- leaf-wise specialization
- port XGBoost feature ideas into a different tree-growth bias
- high-cardinality distribution features

S6E3-inspired variants:

- tuned `num_leaves`, `learning_rate`, `min_child_samples`
- count ratio features
- KDTree nearest-neighbor target features
- synthetic artifact features
- Optuna-optimized configs

### CatBoost

Useful roles:

- native categorical handling
- ordered target statistics for leak-resistant raw categorical encoding
- strong baseline when many categorical columns exist

S6E3-inspired variants:

- minimal frontend with native `cat_features`
- explicit binned cross-term interactions
- borrowed neural-style features such as digit features and triple tokens
- CatBoost-specific tuning such as `random_strength`, `bagging_temperature`, and
  `l2_leaf_reg`

### YDF

Useful role:

- ultra-regularized diversity model

Notes:

- Very shallow trees, such as `max_depth=2`, may produce smooth prediction
  surfaces.
- Value comes from ensemble diversity, not single-model dominance.

### cuML Random Forest

Useful role:

- GPU-accelerated bagging model

Notes:

- Adds diversity because bagging differs from boosting.
- Requires GPU/RAPIDS-compatible environment.

## Neural and Foundation Model Families

Neural tabular models are optional experiment families. They should be enabled
through `nn` dependencies and explicit configs.

Candidate families:

- Embedding MLP:
  - categorical embeddings plus numeric features
  - dropout and BatchNorm/LayerNorm
  - multiple seeds, batch sizes, learning rates, and schedulers for diversity
- Pair Embedding MLP:
  - explicit ordered feature-pair embeddings
  - useful for second-order interactions
- Enhanced Feature MLP:
  - MLP fed with tree-inspired transforms such as frequency, rank, log1p, sqrt,
    and reciprocal features
- RealMLP:
  - PLR numeric features
  - robust scaling
  - label smoothing
  - internal ensemble
- GraphSAGE or KNN-GNN:
  - rows as nodes
  - KNN graph in feature space
  - neighborhood aggregation
- FT-Transformer:
  - tokenizes features and uses self-attention across feature tokens
- TabTransformer:
  - attention over categorical embeddings
  - numeric features handled by MLP head
- TabM:
  - PLR embeddings
  - multiplicative/bilinear interactions
  - implicit basis-component ensemble
- TabICL:
  - in-context tabular foundation model
  - no gradient updates
  - context-length and sampling constraints must be recorded
- GANDALF:
  - gated feature learning units
  - staged feature filtering
- SELU self-normalizing MLP:
  - SELU and AlphaDropout
  - can combine numeric-as-text features and Benford features
- Tabular ResNet:
  - residual MLP blocks
  - can use manifold projection and cyclical features
- RFF Kernel Network:
  - random Fourier features approximate RBF kernels
  - combine with categorical embeddings and prototype distances
- Denoising Autoencoder:
  - train on original/reference data or train+test unsupervised data
  - use reconstruction error and latent embeddings as features
- FM / FFM / DeepFM / DeepFFM:
  - explicit pairwise interactions
  - field-aware interactions
  - optional deep branch for higher-order terms
- Liquid Neural Network:
  - continuous-time-inspired dynamics
  - experimental, diversity-oriented
- VSN / TFT-style:
  - variable selection through gated residual networks
- TabNet:
  - sparse attention and self-supervised pretraining
- Trompt:
  - prompt-style tabular model with layer-wise supervision
- DANet:
  - differentiable feature abstraction and interaction filtering
- TabPFN:
  - tabular foundation model
  - large-data subsampling strategy must be documented

Rules:

- Neural model families are not default dependencies.
- Every neural run must record architecture, seed, epochs, batch size, device,
  training time, and OOF path.
- Foundation/in-context models must record context sampling strategy.
- Any model trained on train+test unsupervised pretraining must label
  transductive risk.

## Registry Guidance

`configs/registries/model_zoo.yaml` should eventually separate:

- `mvp`: safe default model families
- `s6e3_core`: families expected in a strong tabular workflow
- `s6e3_expanded`: expensive or specialized candidates
- `experimental`: high-risk or niche diversity models

The registry should track:

- model family
- package/dependency group
- CPU/GPU support
- default objective
- supported task types
- required artifact outputs
- known leakage or transductive risks
