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

The history of optimizers follows a clean arc: early methods like AdaGrad gave each parameter its own learning rate (parameters with large gradients take smaller steps, sparse parameters take larger ones), and RMSProp fixed AdaGrad's tendency to shrink learning rates to zero by using an exponential moving average. **Adam** combined the best of both worlds — momentum for direction plus adaptive rates for scale — and became the default optimizer for most of deep learning.

Adam tracks a running mean $m_t$ and a running variance $v_t$ of the gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
$$

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}, \qquad \theta_{t+1} = \theta_t - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$. The bias correction terms ($1 - \beta^t$) compensate for the zero-initialization of the moment estimates during early training.

There was one remaining flaw: in standard Adam, $L_2$ regularization gets scaled by the adaptive learning rate, weakening weight decay for parameters with large gradients. **AdamW** fixes this by decoupling weight decay from the gradient update, applying it directly to the parameters. This simple change measurably improves generalization and makes AdamW the go-to optimizer in modern practice.

Other notable variants include LAMB (layer-wise adaptive rates for large-batch training) and AdaFactor (memory-efficient factorized second moments).

## Which optimizer to choose

A practical decision framework:

* **Default choice**: AdamW with weight decay 0.01-0.1. Works well across most architectures and datasets.
* **Vision (CNNs)**: SGD with momentum 0.9 + cosine schedule often matches or beats Adam with proper tuning.
* **Transformers / NLP**: AdamW is strongly preferred. SGD converges too slowly for attention-based models.
* **Large batch training**: LAMB or LARS for scaling to very large batch sizes (>8K).
* **Memory constrained**: AdaFactor reduces optimizer state memory by ~50%.

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

* **Step decay**: Reduce $\eta$ by a factor at specified epochs. Simple but requires manual tuning of milestones.
* **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$. Smooth decay that naturally reaches a minimum at the end of training.
* **Warmup**: Linearly increase $\eta$ from zero over the first few thousand steps to stabilize early training, especially important for large batch sizes and transformers. Without warmup, the initial random gradients can push parameters far from their initialization before the model has learned anything useful.
* **One-cycle policy**: Warmup then cosine decay; often yields faster convergence.

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

**Dropout** randomly sets each hidden unit to zero with probability $p$ during training. At test time, weights are scaled by $(1-p)$. The intuition: if any single neuron might be randomly silenced, the network cannot rely on any one neuron too heavily. It must spread the knowledge across many neurons, creating an implicit ensemble over exponentially many sub-networks.

**Batch normalization** normalizes activations within each mini-batch:

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

Then came **random search**: instead of an exhaustive grid, sample hyperparameters randomly from specified ranges. This sounds worse but is actually much better, because not all hyperparameters are equally important. If the learning rate matters far more than dropout rate, random search explores more distinct learning rates in the same budget.

**Bayesian optimization** got even cleverer: it builds a surrogate model (e.g., Gaussian process) of the validation loss as a function of the hyperparameters, then selects the next configuration to try by maximizing expected improvement. Each experiment teaches the model something about the hyperparameter landscape, making each subsequent choice more informed.

**Population-based training** takes a different approach entirely: run many configurations in parallel, and periodically replace the worst performers with perturbed copies of the best. The hyperparameters evolve during training, adapting to different phases of optimization.

## Initialization

Before training begins, the weights must start somewhere. This seemingly mundane choice can make or break training. If weights are too large, activations explode and gradients blow up. If too small, activations shrink to zero and gradients vanish. The goal is to keep the variance of activations and gradients approximately constant across layers.

* **Xavier/Glorot**: $W \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$. Designed for linear or sigmoid activations.
* **He/Kaiming**: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$. Designed for ReLU activations, which kill half the signal (negative values become zero), so the variance needs to be doubled to compensate.

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

* **Start with a learning rate finder**: Sweep the learning rate from very small to very large over one epoch, plotting loss vs. learning rate. Choose a rate just below where the loss starts increasing.
* **Monitor both train and validation loss**: A growing gap signals overfitting; add regularization or reduce model capacity.
* **Use mixed precision training**: FP16 computation with FP32 master weights accelerates training 2-3x on modern GPUs with minimal accuracy impact.
* **Gradient clipping**: Cap the gradient norm to prevent explosion, especially important for RNNs and transformers: $\hat{g} = g \cdot \min(1, c / \|g\|)$.

## Big Ideas

* The learning rate is arguably the single most important hyperparameter — too large and training diverges, too small and nothing interesting happens, and changing it during training is almost always better than keeping it fixed.
* Dropout's real power is not that it randomly kills neurons, but that it forces the network to distribute knowledge across many pathways — preventing any single neuron from becoming irreplaceable.
* Weight initialization is not a detail: starting with weights that are too large or too small determines whether gradients survive long enough to teach the network anything at all.
* Random search beats grid search not because randomness is magical, but because most hyperparameters are unimportant — random search naturally explores more of what matters.

## What Comes Next

Armed with an optimizer, a regularizer, and a learning rate schedule, you can now train a network that actually generalizes. But all the lessons so far have treated images as flat vectors — a 28x28 image is just 784 numbers in a row, with no notion of which pixels are neighbors. The next lesson introduces convolutional neural networks, which bake the spatial structure of images directly into the architecture. The same ideas of weights, gradients, and backpropagation still apply — but now the network knows that nearby pixels are more likely to be related than distant ones, and it exploits that knowledge to achieve results no flat MLP can match.

## Check Your Understanding

1. Adam adapts the learning rate for each parameter individually based on the history of its gradients. Why might this be helpful for a network where some layers are updated frequently and others rarely?
2. Batch normalization normalizes activations within each mini-batch during training, but uses running statistics at test time. What could go wrong if you forgot to switch to running statistics at test time?
3. Two networks are trained on the same data: one with heavy dropout, one without. The one without dropout achieves lower training loss. Which one would you expect to perform better on new data, and why?

## Challenge

The "learning rate finder" heuristic sweeps the learning rate exponentially from a very small value to a very large one and plots the loss. The learning rate that minimizes the loss is supposed to be a good starting point. Design a theoretical argument for why this heuristic works — and then design an experiment to find a case where it fails. What property of the loss landscape or the data distribution would cause the optimal training learning rate to differ substantially from what the finder suggests?

