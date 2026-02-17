# Decision Trees and Ensemble Methods

Tree-based methods remain strong baselines for structured/tabular data because they capture nonlinear interactions with minimal feature engineering.

## Decision trees
A decision tree recursively partitions feature space to reduce impurity.

- **Gini impurity**:
$$
G(S)=1-\sum_{k=1}^{K}p_k^2
$$
- **Entropy**:
$$
H(S)=-\sum_{k=1}^{K}p_k\log p_k
$$

Trees are interpretable, but deep trees can overfit.

## Ensemble methods
- **Random forest (bagging):** averages many decorrelated trees to reduce variance.
- **AdaBoost / gradient boosting:** builds trees sequentially, emphasizing previous errors.
- **XGBoost-style systems:** optimized gradient boosting with regularization and strong engineering.

## Bias-variance intuition
- Too-simple models: high bias, underfitting.
- Too-complex single trees: high variance, overfitting.
- Ensembles: often improve generalization by balancing both.

## Interactive simulations
[[simulation aml-tree-split-impurity]]

[[simulation aml-tree-ensemble-xor]]

## Model selection notes
- Prefer cross-validation for small tabular datasets.
- Tune `max_depth`, `min_samples_leaf`, `n_estimators`, and learning rate.
- Use calibration checks when probabilities are used for decisions.
