# Analysis and Theory of Deep Learning

## The puzzle of generalization

Here's the central mystery of deep learning. Classical learning theory says: if your model has more parameters than training examples, it will memorize the data and fail on new examples. This has been the bedrock of statistics for a century. Then along comes a neural network with 100 million parameters, trained on 50,000 images, and it *generalizes beautifully*. It has more than enough capacity to memorize every training example -- and it *does* memorize them -- yet it still performs well on data it's never seen. Why?

Understanding this contradiction is one of the deepest open problems in machine learning.

## Double descent

For a hundred years statistics told us: more parameters than data points means guaranteed overfitting. Then deep learning shattered that rule. The **double descent** curve is the field's most jaw-dropping plot: test error goes down, then **up** (classic overfitting), then **down again** as you keep adding parameters past the interpolation threshold.

That second descent is where modern deep learning lives. Among all the functions that perfectly fit the training data, gradient descent somehow finds a particularly smooth one. The phenomenon shows up in three settings: model-wise (varying parameters), epoch-wise (varying training time), and sample-wise (varying data).

[[simulation adl-double-descent]]

## The neural tangent kernel

In the **infinite-width limit**, something remarkable happens: a neural network trained with gradient descent behaves like kernel regression with a fixed kernel called the **neural tangent kernel**:

$$
\Theta(\mathbf{x}, \mathbf{x}') = \left\langle \nabla_\theta f(\mathbf{x}; \theta_0), \, \nabla_\theta f(\mathbf{x}'; \theta_0) \right\rangle.
$$

In this regime the network becomes a lazy student who never changes its mind about the shape of the world. Training dynamics become linear, convergence to a global minimum is guaranteed, and the result is equivalent to kernel ridge regression.

Why this matters *and* why it's not the whole story: the NTK theory is elegant, but it describes an idealization. Real (finite-width) networks exhibit **feature learning** -- the internal representations themselves evolve during training. That feature learning is believed to be essential for practical success. The lazy regime explains convergence; the rich regime explains performance.

## Loss landscapes

The loss function of a deep network defines a surface in a space with millions of dimensions. What does it look like? The answer is far more forgiving than you'd expect.

**In 10,000 dimensions, almost every critical point is a saddle point, not a local minimum.** A local minimum requires the Hessian to have all positive eigenvalues -- the loss must curve upward in *every* direction simultaneously. In high dimensions, that's astronomically unlikely. At a random critical point, almost all eigenvalues have mixed signs, meaning at least one direction curves downward. This is why gradient descent works at all: it's almost never truly stuck, because there's almost always a direction that leads further downhill.

[[simulation adl-loss-landscape]]

Key properties:
* **Local minima vs saddle points**: Most critical points are saddles. The loss at local minima (when they exist) tends to be close to the global minimum.
* **Mode connectivity**: Different solutions found by SGD are often connected by paths of nearly constant loss. The valleys form a connected web.
* **Sharpness and generalization**: Flatter minima tend to generalize better. Sharp minima are fragile -- tiny weight perturbations cause large loss changes, and that fragility usually means the solution doesn't transfer to new data.

## Adversarial robustness

**Adversarial examples** reveal something disturbing: imperceptibly small changes to an input can cause confident misclassification. A picture of a panda, modified by adding noise invisible to the human eye, gets classified as a gibbon with 99% confidence. Nature is sneaky.

The simplest attack is **FGSM**, which perturbs the input in the direction that increases the loss most rapidly:

$$
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y; \theta)).
$$

Stronger attacks (PGD, C&W) use iterative optimization to find the smallest perturbation that causes misclassification.

**Adversarial training** fights back by including worst-case perturbations in the training set:

$$
\min_\theta \mathbb{E}_{(x,y)} \left[\max_{\|\delta\| \leq \epsilon} \mathcal{L}(x + \delta, y; \theta)\right].
$$

A fundamental trade-off exists: robustness typically comes at the cost of reduced clean accuracy. The network must allocate capacity to defending against worst-case perturbations instead of optimizing average-case performance.

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

* **MC Dropout**: Multiple forward passes with dropout enabled at test time; variance across predictions estimates uncertainty.
* **Deep ensembles**: Train multiple models with different initializations; disagreement indicates uncertainty. Simple and effective, but expensive.
* **Temperature scaling**: Post-hoc calibration by dividing logits by a learned temperature $T$.

## Interpretability

Understanding what networks learn -- and why they make specific decisions -- is critical for deploying them responsibly:

* **Gradient-based methods** (saliency maps, Grad-CAM): Highlight input regions that most affect the output.
* **Feature visualization**: Optimize an input to maximally activate a specific neuron, revealing the patterns it responds to.
* **Mechanistic interpretability**: Reverse-engineer the learned circuits inside trained networks. Transformer language models develop **induction heads** -- attention patterns that copy previously seen sequences -- which compose in specific ways to implement in-context learning. Networks learn interpretable, reusable subroutines.

## Big Ideas

* Double descent is a crack in classical statistics: overparameterized models that interpolate the training data can generalize well, which means "more parameters than data" is not the disaster a century of theory predicted.
* In high dimensions, local minima are nearly extinct -- almost all critical points are saddle points, which is why gradient descent works even though nobody guaranteed it would.
* Adversarial examples aren't a bug to be patched; they reveal that networks solve a different problem than we thought -- one that has the right answers on the training distribution but is geometrically fragile.

## What Comes Next

This lesson closes the arc. You started with a single neuron learning to draw a straight line, then stacked them into networks that classify, segment, generate, and reason over sequences. Along the way you met the craft of optimization, the architectural innovations of convolutions and attention, and the generative frameworks of VAEs and GANs. This final lesson asked what guarantees you can make about any of it -- and found that honest theory is still catching up to practice.

The honest summary is this: deep learning works far better than our theories predict, and understanding why is one of the most important open problems in science. The tools here -- neural networks, backpropagation, attention, generative models -- are not the final word. They are the current best answers to the question of how to extract structure from data. The next generation of ideas will likely come from people who know these foundations deeply enough to question them.
