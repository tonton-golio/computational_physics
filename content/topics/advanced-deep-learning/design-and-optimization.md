# Design and Optimization of Deep Learning

## The optimization problem

Training a deep network means finding the parameter values that make the loss as small as possible. Imagine standing on a vast mountain range in the dark, and all you have is a flashlight that shows you the slope under your feet. You take a step downhill, check the slope again, and repeat. That is gradient descent. The question is: how big should each step be, and should you remember which direction you have been going?

Formally, we minimize a loss function $\mathcal{L}(\theta)$ over parameters $\theta$. The choice of optimizer profoundly affects both how fast you get to a good solution and how good that solution ultimately is.

## Stochastic gradient descent

**Stochastic gradient descent** (SGD) updates parameters using a mini-batch gradient estimate:

$$
\theta_{t+1} = \theta_t - \eta \, \hat{g}_t, \qquad \hat{g}_t = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}_i(\theta_t).
$$

Why use a mini-batch instead of the full dataset? Two reasons: it is faster (you update after seeing 100 examples instead of 60,000), and the noise in the mini-batch gradient actually helps escape shallow local minima — like shaking a ball on a bumpy surface helps it roll into deeper valleys.

**SGD with momentum** accumulates a velocity term that smooths oscillations:

$$
v_{t+1} = \mu v_t + \hat{g}_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

Momentum helps the optimizer build speed in consistent gradient directions while dampening oscillations in noisy directions. Think of a heavy ball rolling downhill: it picks up speed in the consistent downhill direction and resists being deflected by small bumps. A typical value is $\mu = 0.9$.

[[simulation adl-optimizer-trajectories]]

## Adaptive learning rate methods

The next breakthrough was giving each parameter its own personalized learning rate. Parameters with consistently large gradients should take smaller steps (they are already learning fast), while parameters with small or sparse gradients should take larger steps (they need more encouragement).

**Adam** (Kingma and Ba, 2015) combines momentum with per-parameter adaptive learning rates:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The bias correction terms ($1 - \beta^t$) compensate for the zero-initialization of the moment estimates during early training.

**Variants**:

- **AdamW**: Decouples weight decay from the gradient update. In standard Adam, $L_2$ regularization is scaled by the adaptive learning rate, weakening its effect for parameters with large gradients. AdamW applies weight decay directly, which improves generalization.
- **LAMB**: Layer-wise adaptive learning rates for large-batch training. Scales the update by the ratio of parameter norm to update norm for each layer.
- **AdaFactor**: Memory-efficient by factorizing the second-moment matrix using row and column statistics instead of storing the full matrix.

## Which optimizer to choose

A practical decision framework:

- **Default choice**: AdamW with weight decay 0.01-0.1. Works well across most architectures and datasets.
- **Vision (CNNs)**: SGD with momentum 0.9 + cosine schedule often matches or beats Adam with proper tuning.
- **Transformers / NLP**: AdamW is strongly preferred. SGD converges too slowly for attention-based models.
- **Large batch training**: LAMB or LARS for scaling to very large batch sizes (>8K).
- **Memory constrained**: AdaFactor reduces optimizer state memory by ~50%.

```python
# AdamW with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# SGD with momentum for vision
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
```
<!--code-toggle-->
```pseudocode
// AdamW with weight decay
optimizer = ADAMW(model.parameters(), lr=3e-4, weight_decay=0.01)

// SGD with momentum for vision
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
```

## Learning rate schedules

The learning rate is arguably the single most important hyperparameter. Too high and training diverges; too low and it crawls. The best strategy is to change it during training:

- **Step decay**: Reduce $\eta$ by a factor at specified epochs. Simple but requires manual tuning of milestones.
- **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$. Smooth decay that naturally reaches a minimum at the end of training.
- **Warmup**: Linearly increase $\eta$ from zero over the first few thousand steps to stabilize early training, especially important for large batch sizes and transformers. Without warmup, the initial random gradients can push parameters far from their initialization before the model has learned anything useful.
- **One-cycle policy**: Warmup then cosine decay; often yields faster convergence.

[[simulation adl-lr-schedule-comparison]]

```python
# Cosine annealing with warmup
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# One-cycle policy
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
```
<!--code-toggle-->
```pseudocode
// Cosine annealing
scheduler = COSINE_ANNEALING(optimizer, T_max=100, eta_min=1e-6)

// One-cycle policy
scheduler = ONE_CYCLE(optimizer, max_lr=0.01, total_steps=1000)
```

## Regularization techniques

Deep networks have an enormous capacity to memorize training data. Regularization techniques fight overfitting by constraining what the network can learn, forcing it to find simpler patterns that generalize.

**Dropout** (Srivastava et al., 2014) randomly sets each hidden unit to zero with probability $p$ during training. At test time, weights are scaled by $(1-p)$. The intuition: if any single neuron might be randomly silenced, the network cannot rely on any one neuron too heavily. It must spread the knowledge across many neurons, creating an implicit ensemble over exponentially many sub-networks.

**Batch normalization** (Ioffe and Szegedy, 2015) normalizes activations within each mini-batch:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta,
$$

where $\mu_B$ and $\sigma_B^2$ are the batch mean and variance, and $\gamma, \beta$ are learnable parameters. Benefits include smoother loss landscapes, faster training, and reduced sensitivity to initialization.

**Layer normalization** normalizes across features instead of across the batch, making it suitable for transformers and variable-length sequences where batch statistics are unreliable.

**Data augmentation** enlarges the effective training set through transformations (rotations, crops, color jitter, mixup). For images, augmentation is one of the most effective regularizers — it forces the network to learn that a cat rotated 15 degrees is still a cat.

**Weight decay** adds an $L_2$ penalty $\lambda \|\theta\|^2$ to the loss, shrinking parameters toward zero and discouraging overfitting.

[[simulation adl-regularization-effects]]

## What if we didn't have regularization?

Without any regularization, a sufficiently large network would memorize every training example perfectly — including the noise. It would achieve 100% training accuracy and terrible test accuracy. The network would learn that "image #3,742 maps to label 7" rather than learning what the digit 7 actually looks like. Regularization forces the network to find the *pattern* instead of memorizing the *data*.

## The hyperparameter search: a detective story

You have built a network, picked an optimizer, and added some regularization. Now you face the hardest question: what learning rate? How much dropout? How many layers? This is **hyperparameter tuning**, and for years it was the most frustrating part of deep learning.

The first attempt was **grid search**: try every combination on a grid. Learning rate in $\{0.1, 0.01, 0.001\}$, dropout in $\{0.3, 0.5\}$, layers in $\{2, 4\}$. That is $3 \times 2 \times 2 = 12$ experiments. But add a few more hyperparameters and the grid explodes exponentially. Grid search was hopeless.

Then came **random search** (Bergstra and Bengio, 2012): instead of an exhaustive grid, sample hyperparameters randomly from specified ranges. This sounds worse but is actually much better, because not all hyperparameters are equally important. If the learning rate matters far more than dropout rate, random search explores more distinct learning rates in the same budget.

**Bayesian optimization** got even cleverer: it builds a surrogate model (e.g., Gaussian process) of the validation loss as a function of the hyperparameters, then selects the next configuration to try by maximizing expected improvement. Each experiment teaches the model something about the hyperparameter landscape, making each subsequent choice more informed.

**Population-based training** takes a different approach entirely: run many configurations in parallel, and periodically replace the worst performers with perturbed copies of the best. The hyperparameters evolve during training, adapting to different phases of optimization.

## Initialization

Before training begins, the weights must start somewhere. This seemingly mundane choice can make or break training. If weights are too large, activations explode and gradients blow up. If too small, activations shrink to zero and gradients vanish. The goal is to keep the variance of activations and gradients approximately constant across layers.

- **Xavier/Glorot**: $W \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$. Designed for linear or sigmoid activations.
- **He/Kaiming**: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$. Designed for ReLU activations, which kill half the signal (negative values become zero), so the variance needs to be doubled to compensate.

```python
# He initialization for ReLU networks
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Xavier initialization for sigmoid/tanh
nn.init.xavier_normal_(layer.weight)
```
<!--code-toggle-->
```pseudocode
// He initialization for ReLU networks
KAIMING_INIT(layer.weight, mode="fan_in", activation="relu")

// Xavier initialization for sigmoid/tanh
XAVIER_INIT(layer.weight)
```

## Practical tips

- **Start with a learning rate finder**: Sweep the learning rate from very small to very large over one epoch, plotting loss vs. learning rate. Choose a rate just below where the loss starts increasing.
- **Monitor both train and validation loss**: A growing gap signals overfitting; add regularization or reduce model capacity.
- **Use mixed precision training**: FP16 computation with FP32 master weights accelerates training 2-3x on modern GPUs with minimal accuracy impact.
- **Gradient clipping**: Cap the gradient norm to prevent explosion, especially important for RNNs and transformers: $\hat{g} = g \cdot \min(1, c / \|g\|)$.

## Further reading

- Kingma, D.P. and Ba, J. (2015). *Adam: A Method for Stochastic Optimization*. The paper that introduced the most widely used optimizer in deep learning.
- Loshchilov, I. and Hutter, F. (2019). *Decoupled Weight Decay Regularization*. The AdamW paper, explaining why weight decay should be decoupled from the gradient update.
