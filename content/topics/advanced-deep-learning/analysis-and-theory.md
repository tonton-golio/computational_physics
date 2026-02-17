# Analysis and Theory of Deep Learning

## The puzzle of generalization

Classical learning theory predicts that models with more parameters than training examples should overfit catastrophically. Yet deep networks with millions of parameters generalize well even when they can perfectly memorize the training data. Understanding this contradiction is a central challenge in deep learning theory.

## Generalization bounds

**PAC-Bayes bounds** provide some of the tightest generalization guarantees for neural networks. For a posterior distribution $Q$ over hypotheses and a prior $P$:

$$
\mathbb{E}_{h \sim Q}[\mathcal{L}(h)] \leq \mathbb{E}_{h \sim Q}[\hat{\mathcal{L}}(h)] + \sqrt{\frac{D_{\text{KL}}(Q \| P) + \ln(n/\delta)}{2n}},
$$

where $\hat{\mathcal{L}}$ is the empirical loss, $n$ is the number of training examples, and $\delta$ is the failure probability. The bound favors posteriors that are both accurate and close to the prior.

**Rademacher complexity** measures the ability of a function class to fit random labels:

$$
\mathcal{R}_n(\mathcal{F}) = \mathbb{E}\left[\sup_{f \in \mathcal{F}} \frac{1}{n} \sum_{i=1}^{n} \sigma_i f(x_i)\right],
$$

where $\sigma_i \in \{-1, +1\}$ are random signs. Lower Rademacher complexity implies better generalization.

## Double descent

The **double descent** phenomenon (Belkin et al., 2019) shows that the test error follows a U-shaped curve in the underparameterized regime, peaks at the interpolation threshold (where the model just fits the training data exactly), and then *decreases again* as the model becomes increasingly overparameterized.

This challenges the classical bias-variance trade-off and suggests that overparameterization acts as an implicit regularizer. The phenomenon has been observed in:

- Model-wise double descent: varying the number of parameters.
- Epoch-wise double descent: varying training time.
- Sample-wise double descent: varying the number of training examples.

## The neural tangent kernel (NTK)

In the **infinite-width limit**, a neural network trained with gradient descent behaves like kernel regression with a fixed kernel called the **neural tangent kernel**:

$$
\Theta(\mathbf{x}, \mathbf{x}') = \left\langle \nabla_\theta f(\mathbf{x}; \theta_0), \, \nabla_\theta f(\mathbf{x}'; \theta_0) \right\rangle,
$$

where $\theta_0$ are the initial parameters. In this regime:

- The NTK remains approximately constant during training (**lazy training**).
- Training dynamics become linear, and convergence to a global minimum is guaranteed.
- The trained network is equivalent to kernel ridge regression with the NTK.

**Limitations**: the NTK theory describes an idealization. Finite-width networks exhibit **feature learning**, where the kernel itself evolves during training. This feature learning is believed to be essential for the practical success of deep learning.

## Loss landscapes

The loss function of a deep network defines a high-dimensional surface over the parameter space. Key properties:

- **Local minima vs saddle points**: in high dimensions, most critical points are saddle points rather than local minima. The loss at local minima tends to be close to the global minimum.
- **Mode connectivity**: different solutions found by SGD are often connected by paths of nearly constant loss (**linear mode connectivity**).
- **Sharpness and generalization**: flatter minima tend to generalize better than sharp minima. This motivates techniques like sharpness-aware minimization (SAM).

[[simulation adl-pca-demo]]

## Adversarial robustness

**Adversarial examples** are imperceptibly perturbed inputs that cause confident misclassification:

$$
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x, y; \theta)).
$$

This is the **fast gradient sign method** (FGSM). Stronger attacks (PGD, C&W) use iterative optimization.

**Adversarial training** augments the training set with adversarial examples, solving:

$$
\min_\theta \mathbb{E}_{(x,y)} \left[\max_{\|\delta\| \leq \epsilon} \mathcal{L}(x + \delta, y; \theta)\right].
$$

A fundamental trade-off exists: robustness to adversarial perturbations typically comes at the cost of reduced clean accuracy.

## Uncertainty quantification

Neural networks are often overconfident in their predictions. Methods for calibrating uncertainty:

- **MC Dropout**: run multiple forward passes with dropout enabled; the variance across predictions estimates uncertainty.
- **Deep ensembles**: train multiple models with different initializations; disagreement indicates uncertainty.
- **Temperature scaling**: post-hoc calibration by dividing logits by a learned temperature $T$.
- **Bayesian neural networks**: place a prior over weights and perform approximate posterior inference (variational inference, MCMC).

## Interpretability

Understanding what deep networks learn:

- **Gradient-based methods** (saliency maps, Grad-CAM): highlight input regions that most affect the output.
- **Feature visualization**: optimize an input to maximally activate a specific neuron or layer.
- **Probing classifiers**: train simple models on intermediate representations to test what information is encoded.
- **Mechanistic interpretability**: reverse-engineer the computations performed by individual circuits within the network.
