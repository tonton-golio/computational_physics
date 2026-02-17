# Tikhonov Regularization

Least squares alone is often too optimistic for inverse problems.
Noise and poor conditioning can make the solution unstable or physically unrealistic.

---

## From Least Squares to Tikhonov

For a linear forward model

$$
\mathbf{d}=\mathbf{Gm},
$$

plain least squares minimizes

$$
E(\mathbf{m})=\|\mathbf{d}-\mathbf{Gm}\|^2.
$$

Tikhonov regularization adds a penalty on model complexity:

$$
E_\epsilon(\mathbf{m})=\|\mathbf{d}-\mathbf{Gm}\|^2+\epsilon^2\|\mathbf{m}\|^2.
$$

The regularized solution is

$$
\hat{\mathbf{m}}=(\mathbf{G}^T\mathbf{G}+\epsilon^2\mathbf{I})^{-1}\mathbf{G}^T\mathbf{d}.
$$

---

## Choosing the Regularization Strength

The parameter $\epsilon$ controls the bias-variance trade-off:

- small $\epsilon$: data fit is strong, noise amplification risk is high
- large $\epsilon$: solution is smoother and more stable, but can underfit

In practice, you sweep a range of $\epsilon$ values and inspect stability, residuals, and physical plausibility.

[[simulation tikhonov-regularization]]

---

## Optimization View

The same objective can be optimized iteratively.
Gradient-based methods are useful for large models and nonlinear extensions.

[[simulation steepest-descent]]

For the quadratic objective above, steepest descent converges to the same regularized minimizer when steps are chosen properly.

---

## Why Regularization Is Physically Meaningful

Regularization encodes prior structure:

- smoothness
- bounded energy
- sparse or low-complexity models

This mirrors how coarse-grid parameterizations are used in climate and Earth-system models.

[[figure climate-grid]]

A Bayesian interpretation is also common: Tikhonov is equivalent to a Gaussian prior on model parameters.

[[figure gaussian-process]]

---

## Weighted Formulation (Data and Model Covariance)

If data covariance is $\mathbf{C}_D$ and model covariance is $\mathbf{C}_M$, define whitening transforms:

$$
\mathbf{V}^T\mathbf{V}=\mathbf{C}_D^{-1}, \qquad \mathbf{W}^T\mathbf{W}=\mathbf{C}_M^{-1}.
$$

Then solve in transformed variables:

$$
\bar{\mathbf{d}}=\mathbf{Vd}, \qquad \bar{\mathbf{m}}=\mathbf{Wm}, \qquad \bar{\mathbf{G}}=\mathbf{VGW}^{-1}.
$$

This makes uncertainty weighting explicit and improves interpretability.

---

## Takeaway

Most practical inverse problems are noisy and ill-conditioned.
Tikhonov regularization is the baseline tool for making inversion stable, interpretable, and computationally tractable.





