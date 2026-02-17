# Design and Optimization of Deep Learning

## Optimization algorithms

Training a deep network means minimizing a loss function $\mathcal{L}(\theta)$ over parameters $\theta$. The choice of optimizer profoundly affects convergence speed and final performance.

**Stochastic gradient descent** (SGD) updates parameters using a mini-batch gradient estimate:

$$
\theta_{t+1} = \theta_t - \eta \, \hat{g}_t, \qquad \hat{g}_t = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}_i(\theta_t).
$$

**SGD with momentum** accumulates a velocity term that smooths oscillations:

$$
v_{t+1} = \mu v_t + \hat{g}_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

## Adaptive learning rate methods

**Adam** (Kingma and Ba, 2015) combines momentum with per-parameter adaptive learning rates:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Variants**:

- **AdamW**: decouples weight decay from the gradient update, which improves generalization.
- **LAMB**: layer-wise adaptive learning rates for large-batch training.
- **AdaFactor**: memory-efficient by factorizing the second-moment matrix.

## Learning rate schedules

The learning rate $\eta$ often varies during training:

- **Step decay**: reduce $\eta$ by a factor at specified epochs.
- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$.
- **Warmup**: linearly increase $\eta$ from zero over the first few thousand steps to stabilize early training.
- **One-cycle policy**: warmup then cosine decay; often yields faster convergence.

## Regularization techniques

**Dropout** (Srivastava et al., 2014) randomly sets each hidden unit to zero with probability $p$ during training. At test time, weights are scaled by $(1-p)$. Dropout acts as an implicit ensemble over exponentially many sub-networks.

**Batch normalization** (Ioffe and Szegedy, 2015) normalizes activations within each mini-batch:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta,
$$

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, and $\gamma, \beta$ are learnable parameters. Benefits include smoother loss landscapes, faster training, and reduced sensitivity to initialization.

**Layer normalization** normalizes across features instead of across the batch, making it suitable for transformers and variable-length sequences.

**Data augmentation** enlarges the effective training set through transformations (rotations, crops, color jitter, mixup). For images, augmentation is one of the most effective regularizers.

**Weight decay** adds an $L_2$ penalty $\lambda \|\theta\|^2$ to the loss, shrinking parameters toward zero and discouraging overfitting.

[[simulation adl-activation-functions]]

## Hyperparameter tuning

Systematic approaches to finding good hyperparameters:

- **Grid search**: exhaustive but exponentially expensive.
- **Random search** (Bergstra and Bengio, 2012): samples hyperparameters randomly; often more efficient than grid search because not all hyperparameters are equally important.
- **Bayesian optimization**: builds a surrogate model (e.g., Gaussian process) of the validation loss and selects the next hyperparameters to maximize expected improvement.
- **Population-based training**: combines random search with online adaptation by periodically replacing poorly performing configurations with perturbed copies of better ones.

## Initialization

Proper weight initialization prevents vanishing or exploding gradients:

- **Xavier/Glorot**: $W \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$. Designed for linear or sigmoid activations.
- **He/Kaiming**: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$. Designed for ReLU activations.

The goal is to keep the variance of activations and gradients approximately constant across layers.
