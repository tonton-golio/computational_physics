# Bayesian Statistics and Multivariate Analysis

## Bayes' Theorem

**Bayesian statistics** is a framework for statistical inference in which beliefs about the probability of an event are updated as new data is obtained. This contrasts with frequentist statistics, which treats probability as a fixed, long-run frequency.

**Bayes' theorem** states that the posterior probability of a hypothesis $H$ given data $D$ is:

$$
P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}
$$

where:
- $P(H|D)$ is the **posterior**: the updated belief after seeing the data.
- $P(D|H)$ is the **likelihood**: the probability of the data given the hypothesis.
- $P(H)$ is the **prior**: the belief before seeing the data.
- $P(D)$ is the **evidence**: a normalization constant.

## Multivariate Analysis (MVA)

**Multivariate analysis** encompasses statistical techniques for data with more than one variable. Key techniques include:

- **Principal component analysis** (PCA): Identifies directions of maximum variance in the data, enabling dimensionality reduction.
- **Factor analysis**: Identifies underlying latent factors that explain correlations between observed variables.
- **Cluster analysis**: Groups similar observations together based on multiple variables.
- **Discriminant analysis**: Classifies observations into groups based on multiple variables.
- **Multivariate regression**: Models relationships between multiple independent variables and a dependent variable.

## The Linear Fisher Discriminant

The **Fisher discriminant** finds the linear combination of features that best separates two classes. The discriminant direction is:

$$
\mathbf{w} = S^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)
$$

where $\boldsymbol{\mu}_0$ and $\boldsymbol{\mu}_1$ are the class means and $S$ is the **pooled within-class covariance matrix**:

$$
S = \frac{1}{n}(S_0 + S_1)
$$

where $S_0$ and $S_1$ are the covariance matrices for each class and $n$ is the total number of observations.

A new observation $\mathbf{x}$ is classified by computing the projection $\mathbf{w}^T \mathbf{x}$ and comparing to a threshold.

This method assumes the classes have normal distributions with identical covariance matrices. When these assumptions are violated, other approaches (e.g., quadratic discriminant analysis or nonlinear classifiers) should be considered.

[[simulation applied-stats-sim-5]]
