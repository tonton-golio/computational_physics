# Machine Learning and Data Analysis

## The Grand Finale

Machine learning is statistics with really good marketing -- and now you know all the tricks inside the black box.

Logistic regression? Maximum likelihood ([PDFs](./probability-density-functions)) applied to classification. Cross-validation? Model comparison from [advanced fitting](./advanced-fitting-calibration). Fisher's discriminant? [ANOVA](./anova)'s variance decomposition in disguise. Everything you've learned is secretly running inside every modern ML algorithm.

ML methods handle three core tasks: **classification** (signal vs. background), **regression** (predicting continuous quantities), and **clustering** (discovering structure without labels). **Supervised learning** trains on labeled examples. **Unsupervised learning** finds structure without them.

## Supervised Learning: Classification

Given $\{(\mathbf{x}_i, y_i)\}$ with $y_i \in \{0, 1\}$, a classifier learns a decision boundary. This is the ML version of [hypothesis testing](./hypothesis-testing): is this event signal or background?

### Logistic Regression

$$
P(y=1 | \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}}
$$

Parameters learned by maximizing likelihood -- same MLE principle, different model. Effective when classes are roughly linearly separable. A solid baseline before trying fancier things.

### Decision Trees and Ensembles

**Decision trees** recursively partition feature space. Interpretable -- you can read the rules -- but prone to overfitting. They memorize noise.

Two fixes:

* **Random forests**: average many trees, each trained on a bootstrap sample with random feature subsets. Averaging smooths out the noise.
* **Boosted decision trees** (BDT): build an ensemble sequentially, each new tree targeting the mistakes of the previous ones. Among the most powerful classifiers for tabular data.

## The Fisher Linear Discriminant

The direction that best separates two classes:

$$
\mathbf{w} = S^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)
$$

where $S = (S_0 + S_1)/n$ is the pooled within-class covariance. This is ANOVA's between/within variance ratio, pointing in the optimal direction. Same assumptions (normal classes, equal covariance) and same fallback when they fail (quadratic or nonlinear classifiers).

## Supervised Learning: Regression

Linear regression minimizes squared residuals -- [chi-square](./chi-square-method) without the per-point weighting. **Ridge** and **lasso** add penalties:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_i (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_p^p
$$

* **Lasso** ($p=1$): drives irrelevant coefficients to exactly zero. Automatic feature selection.
* **Ridge** ($p=2$): shrinks all coefficients toward zero. Better when features are correlated.

## Neural Networks

A neural network is just a giant pile of weighted averages with wiggly nonlinear functions in between:

$$
\mathbf{h}^{(l)} = \sigma\bigl(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\bigr), \quad l = 1, \ldots, L
$$

They can approximate any continuous function (universal approximation theorem) and excel when you don't know which feature combinations matter. The price: large training sets, careful regularization, and harder interpretation.

And here is the climax that ties everything together. That penalty term $\lambda\|\mathbf{w}\|^2$ in regularized regression and neural networks? It's mathematically identical to placing a Gaussian prior on $\mathbf{w}$ and doing MAP estimation ([Bayesian statistics](./bayesian-statistics)). **Regularization *is* Bayesian inference.** The regularization strength $\lambda$ is the prior's precision. This isn't an analogy -- it's an exact equivalence. Every regularized model is secretly doing Bayesian inference whether it knows it or not.

The ridge penalty is a Gaussian prior. Lasso is a Laplace prior. The whole framework of [advanced fitting](./advanced-fitting-calibration) -- nuisance parameters, constraint terms, profile likelihoods -- is the same operation in different clothes. Statistics and machine learning aren't separate fields. They're the same field, wearing different hats.

## Unsupervised Learning

No labels -- just data and a search for structure.

**k-means clustering** partitions data into $k$ groups by iterating: assign points to nearest centroid, update centroids. Think ANOVA in reverse -- instead of testing known groups, you *discover* them.

**PCA** finds directions of maximum variance. The principal components are eigenvectors of the covariance matrix (from [introduction](./introduction-concepts)). Project onto the top $d$ to reduce dimensionality while preserving the most information.

**Factor analysis** posits a generative model: observations are linear combinations of hidden factors plus noise. Unlike PCA (which just finds directions), factor analysis asks: what underlying *causes* produce the correlations we see?

## Model Evaluation

The critical concern: **generalization**. A model that memorizes training data but fails on new data is useless -- it learned noise, not signal.

**Cross-validation** estimates generalization error: split into $k$ folds, train on $k-1$, test on the held-out fold. Rotate. Average.

For classifiers, think about what kind of mistakes you care about. **Precision** answers "of the events I flagged, how many were real?" -- it's the purity of your selection. **Recall** answers "of all the real events, how many did I find?" -- it's the efficiency. There's always a tension: a police officer who arrests everyone has perfect recall but terrible precision. One who never arrests anyone has perfect precision (vacuously) but zero recall.

The **ROC curve** plots true positive rate vs. false positive rate as you vary the threshold. The area under it (AUC) summarizes discrimination power in one number.

[[simulation overfitting-carousel]]

[[simulation roc-live]]

**Overfitting** is the gap between training and test performance. Mitigation: regularization, early stopping, ensembles.

> **Challenge.** Explain overfitting: memorizing the textbook word-for-word aces the practice test, but if the exam has new questions, you fail. Understanding beats memorization. One minute.

## Big Ideas

* ML is statistics rebranded and turbocharged by computation. Recognizing the statistical ideas lets you reason about when algorithms work and when they fail.
* Regularization and Bayesian priors are the same operation. Ridge = Gaussian prior, lasso = Laplace prior.
* Cross-validation is empirical model selection: honest generalization estimates from held-out data.
* Overfitting is the fundamental tension between fitting what you have and generalizing to what you haven't seen.

## What Comes Next

You've traveled the full arc: summarizing data, modeling with distributions, propagating uncertainties, fitting models, testing hypotheses, comparing groups, designing experiments, handling hierarchy and time, reasoning Bayesian-style, managing systematics, and connecting it all to machine learning.

The deepest skill isn't any individual method. It's the habit of asking: what is the probabilistic structure of my problem, and which tool honestly answers my question given the data I actually have?

## Check Your Understanding

1. A neural network: 99% training accuracy, 72% validation. A decision tree: 91% training, 89% validation. Which do you deploy, and what does each gap tell you?
2. In what sense is ridge regression a Gaussian prior on coefficients? What prior is lasso? Why does lasso produce sparse solutions?
3. PCA on 50 features: first two components explain 85% of variance. "Always retain enough for 95%." What's wrong with this rule?

## Challenge

You classify particle physics events with a BDT trained on simulation. On real data, performance degrades. Diagnose: simulation-reality mismatch, overfitting to simulation artifacts, feature drift over time. For each, sketch a diagnostic test and a mitigation strategy.
