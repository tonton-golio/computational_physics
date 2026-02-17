# Loss Functions and Optimization

Loss functions define what "good predictions" mean. Optimization algorithms determine whether we can actually reach good parameters efficiently.

## Classification losses
- **Zero-one loss** is useful for evaluation, but not differentiable for gradient-based training.
- **Hinge loss** enforces margins and is common in linear margin methods.
- **Binary cross-entropy** links naturally to probabilistic outputs.

For binary cross-entropy:
$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{i=1}^{N}\left(y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\right)
$$

## Regression losses
- **MSE** penalizes large errors strongly.
- **MAE** is robust to outliers but non-smooth at zero.
- **Huber** balances MSE and MAE behavior.

## Optimization dynamics
Gradient descent updates parameters using local slope:
$$
\theta_{t+1}=\theta_t-\eta \nabla_{\theta}\mathcal{L}(\theta_t)
$$
where $\eta$ is the learning rate.

Modern optimizers (momentum, RMSProp, Adam) improve stability and speed but still require careful tuning.

## Validation and overfitting
Always separate:
1. Training set for fitting parameters.
2. Validation set for model and hyperparameter selection.
3. Test set for final unbiased performance estimate.

For time series, use chronological splits (rolling or expanding windows) instead of random k-fold to avoid leakage.

## Interactive simulations
[[simulation aml-loss-functions]]

[[simulation aml-loss-landscape]]

[[simulation aml-validation-split]]

## Practical checklist
- Start with a simple baseline model and conservative learning rate.
- Monitor both training and validation metrics.
- Use early stopping and regularization when validation diverges.
- Tune one hyperparameter family at a time before broad sweeps.
