# Analysis and Theory of Deep Learning

## The puzzle of generalization

Here is the central mystery of deep learning. Classical learning theory says: if your model has more parameters than training examples, it will memorize the data and fail on new examples. This is the fundamental bias-variance tradeoff, and it has been the bedrock of statistics for a century. Then along comes a neural network with 100 million parameters, trained on 50,000 images, and it *generalizes beautifully*. It has more than enough capacity to memorize every training example — and it does memorize them — yet it still performs well on data it has never seen. Why?

Understanding this contradiction is one of the deepest open problems in machine learning.

## Generalization bounds

**PAC-Bayes bounds** provide some of the tightest generalization guarantees for neural networks. The idea is to measure how much the trained model differs from what you expected before seeing the data. For a posterior distribution $Q$ over hypotheses and a prior $P$:

$$
\mathbb{E}_{h \sim Q}[\mathcal{L}(h)] \leq \mathbb{E}_{h \sim Q}[\hat{\mathcal{L}}(h)] + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \ln(n/\delta)}{2n}},
$$

where $\hat{\mathcal{L}}$ is the empirical loss, $n$ is the number of training examples, and $\delta$ is the failure probability. The bound favors models that are both accurate on training data and close to the prior — models that did not have to change much to fit the data.

**Rademacher complexity** takes a different approach, measuring the ability of a function class to fit random noise:

$$
\mathcal{R}_n(\mathcal{F}) = \mathbb{E}\left[\sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \sigma_i f(x_i)\right],
$$

where $\sigma_i \in \{-1, +1\}$ are random signs. If your model class can easily fit random labels, it has high Rademacher complexity and weak generalization guarantees. Lower Rademacher complexity implies better generalization.

## Double descent

The **double descent** phenomenon is one of the most surprising empirical discoveries in modern machine learning. As you increase model size, the test error first follows the classic U-shape: it decreases, then increases as the model starts overfitting. But keep going past the point where the model can exactly fit the training data (the interpolation threshold), and something unexpected happens: the test error starts *decreasing again*.

This challenges the classical bias-variance tradeoff and suggests that overparameterization acts as an implicit regularizer. Among all the functions that perfectly fit the training data, gradient descent finds a particularly smooth one. The phenomenon has been observed in three settings:

* Model-wise double descent: varying the number of parameters.
* Epoch-wise double descent: varying training time.
* Sample-wise double descent: varying the number of training examples.

[[simulation adl-double-descent]]

## The neural tangent kernel (NTK)

In the **infinite-width limit**, something remarkable happens: a neural network trained with gradient descent behaves like a simple, well-understood algorithm — kernel regression with a fixed kernel called the **neural tangent kernel**:

$$
\Theta(\mathbf{x}, \mathbf{x}') = \left\langle \nabla_\theta f(\mathbf{x}; \theta_0), \, \nabla_\theta f(\mathbf{x}'; \theta_0) \right\rangle,
$$

where $\theta_0$ are the initial parameters. In this infinite-width regime, the network becomes a lazy student who never changes its mind about the shape of the world. The NTK remains approximately constant during training (**lazy training**), meaning the network adjusts its weights linearly without fundamentally reorganizing its internal representations:

* Training dynamics become linear, and convergence to a global minimum is guaranteed.
* The trained network is equivalent to kernel ridge regression with the NTK.

**Why this matters and why it is not the whole story**: the NTK theory is elegant and provides convergence guarantees, but it describes an idealization. Finite-width networks exhibit **feature learning**, where the internal representations themselves evolve during training. This feature learning — the network discovering new ways to see the data — is believed to be essential for the practical success of deep learning. The lazy regime explains convergence; the rich regime explains performance.

## Loss landscapes

The loss function of a deep network defines a surface in a space with millions of dimensions. What does this surface look like? The answer turns out to be far more forgiving than you might expect.

**In 10,000 dimensions, almost every critical point is a saddle point, not a local minimum.** A local minimum requires the loss to curve upward in *every* direction simultaneously. In high dimensions, this is astronomically unlikely — at a random critical point, roughly half the directions curve up and half curve down, making it a saddle. This is why deep networks can be trained at all: gradient descent is almost never truly stuck, because there is almost always a direction that leads further downhill.

Key properties of the loss landscape:
* **Local minima vs saddle points**: Most critical points are saddles. The loss at local minima (when they exist) tends to be close to the global minimum.
* **Mode connectivity**: Different solutions found by SGD are often connected by paths of nearly constant loss (**linear mode connectivity**). The valleys are not isolated; they form a connected web.
* **Sharpness and generalization**: Flatter minima tend to generalize better than sharp minima. A sharp minimum means tiny perturbations to the parameters cause large changes in the loss — that fragility usually means the solution does not transfer well to new data. This motivates techniques like sharpness-aware minimization (SAM).

[[simulation adl-loss-landscape]]

## What if the loss landscape were full of local minima?

If the loss landscape were riddled with deep, isolated local minima (like potholes on a road), gradient descent would get stuck in whichever pothole it happened to fall into first, and different random initializations would find wildly different (and mostly bad) solutions. The fact that high-dimensional loss landscapes are dominated by saddle points, not local minima, is what makes optimization tractable. The landscape is more like a gentle, undulating terrain with many paths to the valley floor.

## Adversarial robustness

**Adversarial examples** reveal a disturbing property of neural networks: imperceptibly small changes to an input can cause confident misclassification. A picture of a panda, modified by adding noise invisible to the human eye, gets classified as a gibbon with 99% confidence.

The simplest attack is the **fast gradient sign method** (FGSM), which perturbs the input in the direction that increases the loss most rapidly:

$$
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y; \theta)).
$$

Stronger attacks (PGD, C&W) use iterative optimization to find the smallest perturbation that causes misclassification.

**Adversarial training** fights back by augmenting the training set with adversarial examples, solving a minimax problem:

$$
\min_\theta \mathbb{E}_{(x,y)} \left[\max_{\|\delta\| \leq \epsilon} \mathcal{L}(x + \delta, y; \theta)\right].
$$

A fundamental trade-off exists: robustness to adversarial perturbations typically comes at the cost of reduced clean accuracy. The network must allocate some of its capacity to defending against worst-case perturbations instead of optimizing average-case performance.

```python
# FGSM attack
def fgsm_attack(model, x, y, epsilon=0.03):
    x.requires_grad = True
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    return x_adv.clamp(0, 1)
```
<!--code-toggle-->
```pseudocode
FUNCTION fgsm_attack(model, x, y, epsilon=0.03):
    ENABLE_GRADIENTS(x)
    loss = CROSS_ENTROPY(model(x), y)
    BACKWARD(loss)
    x_adv = x + epsilon * SIGN(GRADIENT(x))
    RETURN CLAMP(x_adv, 0, 1)
```

## Uncertainty quantification

Neural networks are often dangerously overconfident. A classifier might assign 99.9% probability to the wrong class. Methods for calibrating uncertainty:

* **MC Dropout**: Run multiple forward passes with dropout enabled at test time; the variance across predictions estimates uncertainty. If the predictions disagree, the model is unsure.
* **Deep ensembles**: Train multiple models with different initializations; disagreement indicates uncertainty. Simple and effective, but expensive.
* **Temperature scaling**: Post-hoc calibration by dividing logits by a learned temperature $T$. Higher temperature softens the predictions.
* **Bayesian neural networks**: Place a prior over weights and perform approximate posterior inference (variational inference, MCMC). The theoretically principled approach, but computationally demanding.

## Interpretability

Understanding what deep networks learn — and why they make specific decisions — is critical for deploying them in high-stakes domains:

* **Gradient-based methods** (saliency maps, Grad-CAM): Highlight input regions that most affect the output. Grad-CAM uses the gradient of the target class score with respect to the final convolutional layer to produce a coarse localization map showing where the network is looking.
* **Feature visualization**: Optimize an input to maximally activate a specific neuron or layer, revealing the patterns each neuron responds to. The resulting images are often hauntingly beautiful and alien.
* **Probing classifiers**: Train simple models on intermediate representations to test what information is encoded at each layer. Does layer 5 know about syntax? Does layer 12 know about sentiment?
* **Mechanistic interpretability**: Reverse-engineer the computations performed by individual circuits within the network, identifying specific algorithmic roles for groups of neurons. The goal is to understand the network as an algorithm, not just a black box.

## Big Ideas

* Double descent is a crack in classical statistics: overparameterized models that interpolate the training data can generalize well, which means "more parameters than data points" is not the disaster a century of theory predicted.
* In high dimensions, local minima are nearly extinct — almost every critical point is a saddle with at least one direction pointing downhill, which is why gradient descent works even though we never guaranteed it would.
* Adversarial examples are not a bug to be patched; they reveal that neural networks are solving a different problem than we thought — one that has the right answers on the training distribution but is fragile in ways that are geometrically inevitable given the decision boundary's shape.
* The NTK regime (infinite width) and the feature learning regime (finite width) are two different theories of the same object — and the success of real deep learning lives entirely in the regime the elegant theory cannot fully explain.

## What Comes Next

This lesson closes the arc of the topic. You started with a single neuron learning to draw a straight line, then stacked them into networks that could classify, segment, generate, and reason over sequences. Along the way you met the practical craft of optimization and regularization, the architectural innovations of convolutions and attention, and the generative frameworks of VAEs and GANs. This final lesson asked what guarantees you can make about any of it — and found that honest theory is still catching up to empirical practice.

The honest summary is this: deep learning works far better than our theories predict, and understanding why is one of the most important open problems in science. The tools here — neural networks, backpropagation, attention, generative models — are not the final word. They are the current best answers to the question of how to extract structure from data. The next generation of ideas will likely come from people who know these foundations deeply enough to question them.

## Check Your Understanding

1. Classical learning theory predicts that a model with more parameters than training examples will overfit. Deep neural networks routinely violate this prediction and still generalize. What does double descent suggest about the mechanism by which overparameterized models generalize, and what property of gradient descent might be responsible?
2. The neural tangent kernel analysis shows that infinitely wide networks converge to global minima and behave like kernel methods. Why is this a partial victory for theory — and what does "feature learning" mean, and why does its absence in the NTK regime matter?
3. Adversarial training makes a model robust to small perturbations by including worst-case perturbations in the training set. Why does this generally reduce accuracy on clean, unperturbed inputs, and what does this trade-off reveal about the geometry of the decision boundary?

## Challenge

Design an experiment to test whether flat minima generalize better than sharp minima. Train the same network multiple times with different optimizers or learning rate schedules — some known to converge to sharp minima (large learning rate, sharp decay), some known to favor flatter minima (small learning rate, stochastic noise, or sharpness-aware minimization). For each run, measure the sharpness of the final minimum (e.g., the largest eigenvalue of the Hessian of the loss, or the loss change under random weight perturbations), and measure the test accuracy. Plot sharpness vs. test accuracy. Does the correlation hold? Does it hold equally well across different architectures and datasets, or does the relationship break down in some regime?
