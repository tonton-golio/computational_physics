# Machine Learning and Data Analysis

## Overview

Machine learning (ML) provides algorithms that learn patterns from data without being explicitly programmed for each case. In experimental physics and data analysis, ML methods complement traditional statistical techniques for tasks like **classification** (separating signal from background), **regression** (predicting continuous quantities), and **clustering** (discovering structure in unlabelled data).

ML algorithms fall into two broad categories. **Supervised learning** trains on labelled examples and predicts labels for new data. **Unsupervised learning** finds structure in data without labels.

## Supervised Learning: Classification

Given training data $\{(\mathbf{x}_i, y_i)\}$ where $y_i \in \{0, 1\}$ labels signal vs. background, a classifier learns a decision boundary in feature space.

**Logistic regression** models the probability of the positive class as

$$
P(y=1 | \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}},
$$

where $\mathbf{w}$ and $b$ are learned by maximizing the likelihood. Despite its simplicity, logistic regression is effective when classes are approximately linearly separable.

**Decision trees** recursively partition feature space by selecting the feature and threshold that best separates classes at each node. They are interpretable but prone to overfitting. **Random forests** reduce overfitting by averaging predictions from many trees, each trained on a bootstrap sample with a random subset of features.

**Boosted decision trees** (BDT) build an ensemble sequentially, with each new tree focusing on the examples misclassified by the previous ones. Gradient boosting (e.g., XGBoost) is among the most powerful classifiers for tabular data and is widely used in particle physics for event selection.

## Supervised Learning: Regression

For predicting continuous targets, the same algorithms adapt. Linear regression minimizes squared residuals, while **ridge** and **lasso** regression add $L_2$ and $L_1$ penalty terms respectively:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_i (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_p^p.
$$

Lasso ($p=1$) performs automatic feature selection by driving irrelevant coefficients to zero. Ridge ($p=2$) handles correlated features more gracefully.

## Neural Networks

A feedforward neural network with $L$ layers computes

$$
\mathbf{h}^{(l)} = \sigma\bigl(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\bigr), \quad l = 1, \ldots, L,
$$

where $\sigma$ is a nonlinear activation function (ReLU, sigmoid, or tanh) and $\mathbf{h}^{(0)} = \mathbf{x}$. The parameters are optimized by **backpropagation** using gradient descent on a loss function.

Neural networks can approximate any continuous function (universal approximation theorem) and excel when feature engineering is difficult. However, they require large training sets, careful regularization, and are less interpretable than tree-based methods.

## Unsupervised Learning

**k-means clustering** partitions data into $k$ groups by iteratively assigning points to the nearest centroid and updating centroids. It requires specifying $k$ in advance and assumes roughly spherical clusters.

**Principal component analysis** (PCA) finds the directions of maximum variance in the data. The principal components are the eigenvectors of the covariance matrix. Projecting onto the top $d$ components reduces dimensionality while preserving the most information.

## Model Evaluation

The critical concern in ML is **generalization**: performance on unseen data, not just the training set.

**Cross-validation** splits the data into $k$ folds, trains on $k-1$, and tests on the held-out fold, rotating through all folds. The average test performance estimates generalization error.

For classifiers, key metrics include:
- **Accuracy**: Fraction of correct predictions.
- **Precision**: $\text{TP} / (\text{TP} + \text{FP})$, the purity of selected events.
- **Recall (sensitivity)**: $\text{TP} / (\text{TP} + \text{FN})$, the efficiency for signal.
- **ROC curve**: Plots true positive rate vs. false positive rate as the decision threshold varies. The area under the curve (AUC) summarizes overall discrimination power.

**Overfitting** occurs when the model memorizes training noise. It is detected by a gap between training and validation performance and mitigated by regularization, early stopping, or ensemble methods.

## Time Series Analysis

Many experiments produce sequential measurements where temporal correlations matter. **Autocorrelation** quantifies how a signal correlates with itself at different time lags:

$$
R(\tau) = \frac{1}{N} \sum_{t=1}^{N-\tau} (x_t - \bar{x})(x_{t+\tau} - \bar{x}).
$$

**Moving averages** smooth noisy signals, while **autoregressive (AR) models** predict the next value from a linear combination of previous values. The **Fourier transform** (covered in the FFT topic) reveals periodic components in the frequency domain.

[[simulation applied-stats-sim-8]]
