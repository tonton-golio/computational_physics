# Design and Optimization of Deep Learning

## The optimization problem

Training a deep network means finding the parameter values that make the loss as small as possible. Imagine standing on a vast mountain range in the dark, and all you have is a flashlight that shows you the slope under your feet. You take a step downhill, check the slope again, repeat. That's gradient descent. The real question is: how big should each step be, and should you remember which direction you've been going?

## Stochastic gradient descent

**Stochastic gradient descent** (SGD) updates parameters using a mini-batch gradient estimate:

$$
\theta_{t+1} = \theta_t - \eta \, \hat{g}_t, \qquad \hat{g}_t = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}_i(\theta_t).
$$

Why mini-batches instead of the full dataset? Two reasons: it's faster (you update after 100 examples instead of 60,000), and the noise in the mini-batch gradient actually *helps* escape shallow local minima -- like shaking a ball on a bumpy surface helps it roll into deeper valleys.

**SGD with momentum** adds a velocity term. Think of a heavy ball rolling downhill: it picks up speed in the consistent downhill direction and resists being deflected by small bumps:

$$
v_{t+1} = \mu v_t + \hat{g}_t, \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}.
$$

Typical $\mu = 0.9$.

## Adaptive learning rate methods

The history of optimizers follows a clean arc. AdaGrad gave each parameter its own learning rate. RMSProp fixed AdaGrad's tendency to shrink rates to zero. **Adam** combined the best of both worlds -- momentum for direction plus adaptive rates for scale -- and became the default for most of deep learning.

And here's the thing about Adam: it's like each parameter having its own personal coach who remembers how fast it's been moving. The coach adjusts the step size automatically -- parameters with consistently large gradients take smaller, more careful steps, while rarely-updated parameters take bolder ones.

There was one remaining flaw: in standard Adam, weight decay gets scaled by the adaptive learning rate, weakening regularization for parameters with large gradients. **AdamW** fixes this by decoupling weight decay from the gradient update. This simple change measurably improves generalization and makes AdamW the go-to optimizer in modern practice.

## Which optimizer to choose

* **Default choice**: AdamW with weight decay 0.01-0.1. Works well across most architectures.
* **Vision (CNNs)**: SGD with momentum 0.9 + cosine schedule often matches or beats Adam with proper tuning.
* **Transformers / NLP**: AdamW is strongly preferred. SGD converges too slowly for attention-based models.
* **Large batch**: LAMB or LARS for scaling to very large batch sizes (>8K).

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

## Initialization

Before training begins, the weights must start somewhere. This seemingly mundane choice can make or break training. If weights are too large, activations explode and gradients blow up. If too small, activations shrink to zero and gradients vanish. The goal is to keep the variance of activations and gradients approximately constant across layers.

* **Xavier/Glorot**: $W \sim \mathcal{N}(0, 2/(n_{\text{in}} + n_{\text{out}}))$. Designed for linear or sigmoid activations.
* **He/Kaiming**: $W \sim \mathcal{N}(0, 2/n_{\text{in}})$. Designed for ReLU, which kills half the signal (negative values become zero), so the variance needs doubling to compensate.

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

## Residual connections

Deep networks suffer from a counterintuitive **degradation problem**: adding more layers can actually *increase* training error, even though the network has strictly more capacity. A deeper network should do at least as well as a shallower one -- the extra layers could just learn the identity function. But in practice, learning the identity is surprisingly hard.

**Residual connections** solve this elegantly. Instead of asking the network to learn the full transformation, you ask it to learn only the *difference* from the identity:

$$
\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l; \theta_l).
$$

It's like telling the network: "If you can't figure out what to do, at least copy what was already there." If the optimal transformation is close to identity, the network only needs to learn a small residual $F \approx 0$, which is far easier than learning the full mapping from scratch.

Skip connections also let the gradient flow directly through the identity path, preventing vanishing gradients even in very deep networks. They show up everywhere -- ResNets, U-Nets, transformers.

## Learning rate schedules

The learning rate is arguably the single most important hyperparameter. Too high and training diverges. Too low and it crawls. The best strategy is to change it during training:

* **Cosine annealing**: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$. Smooth decay that naturally reaches a minimum at the end of training.
* **Warmup**: Linearly increase $\eta$ from zero over the first few thousand steps. Without warmup, the initial random gradients can push parameters far from their initialization before the model has learned anything useful. Essential for large batches and transformers.
* **One-cycle policy**: Warmup then cosine decay; often yields faster convergence.

And here's a trick that saves enormous time: the **learning rate finder**. Sweep the learning rate from very small to very large over one epoch, plotting loss vs. learning rate. Choose a rate just below where the loss starts increasing. Ten minutes of sweeping can save days of guessing.

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

Deep networks have enormous capacity to memorize training data. Regularization forces them to find simpler patterns that generalize.

**Dropout** randomly sets each hidden unit to zero with probability $p$ during training. Here's the beautiful intuition: if any single neuron might be randomly silenced, the network can't rely on any one neuron too heavily. It must spread the knowledge across many neurons, creating an implicit ensemble over exponentially many sub-networks.

**Batch normalization** normalizes activations within each mini-batch:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta.
$$

Benefits: smoother loss landscapes, faster training, reduced sensitivity to initialization. **Layer normalization** does the same across features instead of across the batch, making it the right choice for transformers and variable-length sequences.

**Data augmentation** enlarges the effective training set through transformations -- rotations, crops, color jitter. It forces the network to learn that a cat rotated 15 degrees is still a cat.

**Weight decay** adds an $L_2$ penalty $\lambda \|\theta\|^2$, shrinking parameters toward zero.

## The hyperparameter search

You've built a network, picked an optimizer, added regularization. Now the hardest question: what learning rate? How much dropout? How many layers?

**Grid search** tries every combination on a grid. It was hopeless -- add a few hyperparameters and the grid explodes exponentially.

**Random search** sounds worse but is actually much better, because not all hyperparameters are equally important. If the learning rate matters far more than dropout, random search explores more distinct learning rates in the same budget.

**Bayesian optimization** got even cleverer: it builds a surrogate model of the validation loss as a function of hyperparameters, then picks the next configuration to try by maximizing expected improvement. Each experiment teaches the model something about the landscape.

## Big Ideas

* The learning rate is the single most important hyperparameter -- and changing it during training is almost always better than keeping it fixed.
* Dropout's real power is that it forces the network to distribute knowledge across many pathways, preventing any single neuron from becoming irreplaceable.
* Residual connections reframe the problem: instead of learning a transformation, learn a correction to the identity, which turns out to be much easier.

## What Comes Next

Armed with an optimizer, a regularizer, a learning rate schedule, and residual connections, you can train networks that actually generalize -- even very deep ones. But so far we've treated images as flat vectors -- a 28x28 image is just 784 numbers in a row, with no notion of which pixels are neighbors. The next lesson introduces convolutional neural networks, which bake spatial structure directly into the architecture. The residual connections you just learned will reappear as **ResNet**, enabling 100+ layer networks.
