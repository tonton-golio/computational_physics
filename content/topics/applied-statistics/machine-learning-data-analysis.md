# Machine Learning and Data Analysis

## The Grand Finale

Machine learning is just statistics with really good marketing — and now you know all the tricks inside the black box.

Seriously. Logistic regression? That's maximum likelihood estimation (lesson 2) applied to classification. Regularization? The frequentist cousin of Bayesian priors (lesson 11). Cross-validation? A practical implementation of the model comparison ideas from lesson 12. The Fisher discriminant? ANOVA's variance decomposition (lesson 7) in disguise. Everything you've learned — likelihoods, priors, uncertainty quantification, model selection, error propagation — is secretly running inside every modern ML algorithm.

This section makes those connections explicit and introduces the algorithms that have become essential tools in experimental physics and data analysis. ML methods complement traditional techniques for three core tasks: **classification** (separating signal from background), **regression** (predicting continuous quantities), and **clustering** (discovering structure in unlabelled data).

ML algorithms fall into two broad categories. **Supervised learning** trains on labelled examples and predicts labels for new data. **Unsupervised learning** finds structure without labels.

## Supervised Learning: Classification

Given training data $\{(\mathbf{x}_i, y_i)\}$ where $y_i \in \{0, 1\}$ labels signal vs. background, a classifier learns a decision boundary in feature space. This is the ML version of the hypothesis testing problem from lesson 6: is this event signal or background?

### Logistic Regression

**Logistic regression** models the probability of the positive class as:

$$
P(y=1 | \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}},
$$

where $\mathbf{w}$ and $b$ are learned by maximizing the likelihood — the same MLE principle from lesson 2, just applied to a different model. Despite its simplicity, logistic regression is effective when classes are approximately linearly separable and serves as a useful baseline before trying more complex methods.

### Decision Trees and Ensembles

**Decision trees** recursively partition feature space by selecting the feature and threshold that best separates classes at each node. They are interpretable — you can read off the decision rules — but prone to overfitting. They memorize noise in the training data.

Two ensemble strategies fix this:

- **Random forests** reduce overfitting by averaging predictions from many trees, each trained on a bootstrap sample with a random subset of features. The averaging smooths out the noise that individual trees memorize.
- **Boosted decision trees** (BDT) build an ensemble sequentially, with each new tree focusing on the examples the previous ones got wrong. Gradient boosting (e.g., XGBoost) is among the most powerful classifiers for tabular data and is widely used in particle physics for event selection.

## The Fisher Linear Discriminant

The **Fisher discriminant** finds the single linear combination of features that best separates two classes. It connects directly to the variance decomposition from ANOVA (lesson 7): just as ANOVA separates between-group from within-group variance, Fisher finds the direction that maximizes the ratio of between-class to within-class scatter.

The discriminant direction is:

$$
\mathbf{w} = S^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)
$$

where $\boldsymbol{\mu}_0$ and $\boldsymbol{\mu}_1$ are the class means and $S$ is the **pooled within-class covariance matrix**:

$$
S = \frac{1}{n}(S_0 + S_1)
$$

A new observation $\mathbf{x}$ is classified by projecting onto $\mathbf{w}$ and comparing to a threshold. This method assumes normally distributed classes with equal covariance matrices — the same assumptions that underlie ANOVA. When those assumptions are violated, quadratic discriminant analysis or nonlinear classifiers should be used.

## Supervised Learning: Regression

For predicting continuous targets, the same algorithms adapt. Linear regression minimizes squared residuals (the chi-square method from lesson 5 without the weighting by uncertainties). **Ridge** and **lasso** regression add penalty terms:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_i (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_p^p.
$$

These penalties serve the same purpose as the nuisance parameter constraints in lesson 12 — they prevent the model from over-adapting to noise:

- **Lasso** ($p=1$) performs automatic feature selection by driving irrelevant coefficients to exactly zero. Useful when you suspect many features are unimportant.
- **Ridge** ($p=2$) shrinks all coefficients toward zero without eliminating any. Better when features are correlated.

The penalty $\lambda \|\mathbf{w}\|^2$ is mathematically identical to placing a Gaussian prior on $\mathbf{w}$ and doing MAP estimation (lesson 11). Regularization *is* Bayesian inference. The "regularization strength" $\lambda$ is the prior's precision. This isn't just an analogy — it's an exact mathematical equivalence.

## Neural Networks

A feedforward neural network with $L$ layers computes:

$$
\mathbf{h}^{(l)} = \sigma\bigl(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\bigr), \quad l = 1, \ldots, L,
$$

where $\sigma$ is a nonlinear activation function (ReLU, sigmoid, or tanh) and $\mathbf{h}^{(0)} = \mathbf{x}$. The parameters are optimized by **backpropagation** using gradient descent on a loss function.

Neural networks can approximate any continuous function (universal approximation theorem) and excel when feature engineering is difficult — when you don't know in advance which combinations of inputs matter. The price is that they require large training sets, careful regularization, and produce models that are harder to interpret than tree-based methods.

## Unsupervised Learning

Sometimes you don't have labels — you just have data and want to find structure.

**k-means clustering** partitions data into $k$ groups by iteratively assigning points to the nearest centroid and updating centroids. It requires specifying $k$ in advance and assumes roughly spherical clusters. Think of it as ANOVA in reverse — instead of testing whether known groups differ, you *discover* the groups from the data.

**Principal component analysis** (PCA) finds the directions of maximum variance in the data. The principal components are the eigenvectors of the covariance matrix (which you first encountered in lesson 1). Projecting onto the top $d$ components reduces dimensionality while preserving the most information — useful for visualization and for removing noise before applying other methods.

**Factor analysis** identifies latent (hidden) factors that explain correlations among observed variables. Unlike PCA, which simply finds directions of maximum variance, factor analysis posits a generative model: observations are linear combinations of a smaller number of unobserved factors plus noise. It answers the question: what underlying causes could produce the correlations we see?

## Model Evaluation

The critical concern in ML is **generalization**: how well does the model perform on data it has never seen? A model that memorizes the training set perfectly but fails on new data is useless — it has learned the noise, not the signal.

**Cross-validation** provides a practical estimate of generalization error. Split the data into $k$ folds, train on $k-1$, and test on the held-out fold, rotating through all folds. The average test performance estimates how the model will perform in the wild. This is the model comparison idea from lesson 12, made practical.

For classifiers, key metrics include:

- **Accuracy**: Fraction of correct predictions. Simple but misleading when classes are imbalanced.
- **Precision**: $\text{TP} / (\text{TP} + \text{FP})$ — of the events you selected, how many were actually signal? This is the purity of your selection.
- **Recall (sensitivity)**: $\text{TP} / (\text{TP} + \text{FN})$ — of all the signal events that existed, how many did you find? This is the efficiency.
- **ROC curve**: Plots true positive rate vs. false positive rate as the decision threshold varies. The area under the curve (AUC) summarizes discrimination power in a single number.

**Overfitting** occurs when the model memorizes training noise rather than learning the underlying pattern. It's detected by a gap between training and validation performance. Mitigation strategies include regularization (penalty terms, as in ridge/lasso), early stopping (halting training before over-adaptation), and ensemble methods (averaging over multiple models).

## Where to Go from Here

You've reached the end of this course, and you now have a toolkit that covers the full arc of applied statistics: from summarizing data (lesson 1) through probability models (lesson 2), error analysis (lessons 3-4), model fitting (lesson 5), hypothesis testing (lesson 6), group comparisons (lessons 7-8), hierarchical and longitudinal data (lessons 9-10), Bayesian reasoning (lesson 11), advanced fitting (lesson 12), and now machine learning.

Here are some directions to explore:

- **Deep learning**: Convolutional networks for images, recurrent networks for sequences, transformers for everything else. The principles of regularization, cross-validation, and uncertainty quantification you've learned here apply directly.
- **Causal inference**: Moving from "X is correlated with Y" to "X causes Y." Randomized experiments (lesson 8) are the gold standard; when you can't randomize, there are methods like instrumental variables and regression discontinuity.
- **Bayesian computation**: Markov Chain Monte Carlo (MCMC) and variational inference for problems where the posterior can't be computed analytically.
- **High-dimensional statistics**: When you have more features than observations, classical methods break down. Sparse methods, random projections, and compressed sensing fill the gap.

The tools in this course won't just help you analyze data — they'll help you think clearly about a noisy, uncertain world. That's the real skill, and it's one that transfers everywhere.

> **Challenge.** Explain overfitting to a friend. Use this analogy: memorizing the textbook word-for-word lets you ace the practice test, but if the exam has different questions, you fail. A student who *understands* the material can handle both. One minute.
