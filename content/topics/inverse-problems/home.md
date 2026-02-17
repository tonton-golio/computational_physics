# Inverse Problems

## Course overview

Inverse problems ask a fundamental question: **what hidden system produced the data we measured?** You observe effects, then infer causes. This module develops the mathematical and computational tools needed to answer that question reliably, covering information theory, regularization, linear tomography, and Monte Carlo inversion.

- Formulating inverse problems as parameter estimation from indirect measurements.
- Understanding ill-posedness: non-uniqueness, instability, and noise sensitivity.
- Regularization techniques that make inversion stable and physically meaningful.
- Uncertainty quantification through Bayesian and sampling-based approaches.

## Why this topic matters

- Medical imaging (MRI, SPECT), seismic exploration, and climate model calibration all rely on solving inverse problems.
- Real-world measurements are noisy and incomplete, so naive inversion fails without regularization.
- Quantifying uncertainty in inferred models is essential for scientific credibility and decision-making.
- Inverse methods connect linear algebra, optimization, probability, and physics into a unified computational workflow.

## Key mathematical ideas

- Forward models mapping parameters to observables: $\mathbf{d} = g(\mathbf{m})$.
- Hadamard well-posedness conditions and their failure in inverse settings.
- Shannon entropy and Kullback-Leibler divergence for measuring information content.
- Tikhonov regularization and the bias-variance trade-off.
- Linear tomography: ray-based sensitivity matrices and regularized inversion.
- Monte Carlo integration and Markov Chain Monte Carlo (MCMC) for posterior sampling.

## Prerequisites

- Linear algebra: matrix operations, eigenvalues, least squares.
- Calculus: derivatives, integrals, Taylor series.
- Probability and statistics: distributions, Bayes' theorem, expectation.
- Programming in Python with NumPy.

## Learning trajectory

This module progresses from foundations through regularized linear inversion to nonlinear Monte Carlo methods:

- Introduction to inverse problems: forward models, ill-posedness, and the Bayesian viewpoint.
- Information, entropy, and uncertainty: Shannon entropy and KL divergence.
- Regularization and stable inference: Tikhonov regularization and optimization.
- Linear tomography workflow: ray-based forward modeling, inversion, and resolution analysis.
- Nonlinear inversion and Monte Carlo: rejection sampling, MCMC, and posterior exploration.
- Deep dive: Tikhonov regularization theory, weighted formulations, and covariance.
- Deep dive: linear tomography implementation, sensitivity matrices, and resolution.
