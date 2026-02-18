# Applied Machine Learning

## Course overview

Applied machine learning studies how to design models that generalize well to new data while remaining computationally practical. This module progresses from how models learn, through classical methods and data understanding, to neural network architectures for structured, sequential, and relational data.

## Why this order?

We start where every ML system starts: with loss functions and optimization. Before you can appreciate any model, you need to understand what "learning" actually means — a model adjusts its knobs to make a number (the loss) go down. Once you feel that in your bones, we move to decision trees and ensembles, which are the most intuitive and practically powerful models for structured data. You can literally draw them on a napkin. Next comes dimensionality reduction, because before building fancier models you need to understand what your data looks like in high-dimensional space — and how to squash it down so you can see patterns. With that foundation, we introduce neural networks: the general-purpose function approximators behind modern deep learning. Finally, we offer three taster lessons on specialized architectures — recurrent networks for sequences, graph networks for relational data, and GANs for generative modeling — each of which gets full treatment in the Advanced Deep Learning module.

## What you will learn

- **Loss functions and optimization** — how models learn from data, and why the choice of loss function shapes everything downstream.
- **Decision trees and ensemble methods** — strong, interpretable baselines for structured data that you should always try before reaching for neural networks.
- **Dimensionality reduction** — how to compress and visualize high-dimensional spaces so you can see what your model sees.
- **Neural network fundamentals** — the building blocks (layers, activations, backpropagation, regularization) behind all modern deep learning.
- **Recurrent neural networks** (extension) — extending neural networks to sequential data like time series and language.
- **Graph neural networks** (extension) — extending neural networks to relational data that lives on graphs.
- **Generative adversarial networks** (extension) — a bridge from classical ML to modern deep generative modeling.

The three extension lessons are tasters — they give you the core intuition and one worked idea, then point you to [Advanced Deep Learning](/topics/advanced-deep-learning) for the full treatment.

## Why this topic matters

Machine learning drives modern scientific discovery, from molecular property prediction to climate modeling. Understanding loss functions and optimization dynamics is essential for training any model effectively. Tree ensembles remain the strongest baselines for tabular data across many domains. Dimensionality reduction is critical for exploratory data analysis and feature engineering. Neural networks are the foundation of nearly all modern architectures, and graph neural networks open ML to relational data that cannot be expressed as flat tables.

## Prerequisites

- Linear algebra (vectors, matrices, eigenvalues/eigenvectors).
- Probability and statistics.
- Multivariable calculus (gradients and chain rule).
- Introductory Python or JavaScript-level programming familiarity.

## Relation to other modules

If you have taken Applied Statistics, you will recognize some topics here (decision trees, PCA, model evaluation). This module covers them in greater depth with a focus on ML methodology rather than statistical inference. For deeper coverage of specific architectures (CNNs, transformers, VAEs), see [Advanced Deep Learning](/topics/advanced-deep-learning).
