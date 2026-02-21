# Machine Learning and Data Analysis

## The Grand Finale

Machine learning is just statistics with really good marketing — and now you know all the tricks inside the black box.

Seriously. Logistic regression? That's maximum likelihood estimation ([PDFs](./probability-density-functions)) applied to classification. Regularization? The frequentist cousin of Bayesian priors ([Bayesian statistics](./bayesian-statistics)). Cross-validation? A practical implementation of the model comparison ideas from [advanced fitting](./advanced-fitting-calibration). The Fisher discriminant? [ANOVA](./anova)'s variance decomposition in disguise. Everything you've learned — likelihoods, priors, uncertainty quantification, model selection, error propagation — is secretly running inside every modern ML algorithm.

This section makes those connections explicit and introduces the algorithms that have become essential tools in experimental physics and data analysis. ML methods complement traditional techniques for three core tasks: **classification** (separating signal from background), **regression** (predicting continuous quantities), and **clustering** (discovering structure in unlabelled data).

ML algorithms fall into two broad categories. **Supervised learning** trains on labelled examples and predicts labels for new data. **Unsupervised learning** finds structure without labels.

## Supervised Learning: Classification

Given training data $\{(\mathbf{x}_i, y_i)\}$ where $y_i \in \{0, 1\}$ labels signal vs. background, a classifier learns a decision boundary in feature space. This is the ML version of the [hypothesis testing](./hypothesis-testing) problem: is this event signal or background?

### Logistic Regression

**Logistic regression** models the probability of the positive class as:

$$
P(y=1 | \mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T \mathbf{x} - b}},
$$

where $\mathbf{w}$ and $b$ are learned by maximizing the likelihood — the same MLE principle from [probability density functions](./probability-density-functions), just applied to a different model. Despite its simplicity, logistic regression is effective when classes are approximately linearly separable and serves as a useful baseline before trying more complex methods.

### Decision Trees and Ensembles

**Decision trees** recursively partition feature space by selecting the feature and threshold that best separates classes at each node. They are interpretable — you can read off the decision rules — but prone to overfitting. They memorize noise in the training data.

Two ensemble strategies fix this:

* **Random forests** reduce overfitting by averaging predictions from many trees, each trained on a bootstrap sample with a random subset of features. The averaging smooths out the noise that individual trees memorize.
* **Boosted decision trees** (BDT) build an ensemble sequentially, with each new tree focusing on the examples the previous ones got wrong. Gradient boosting (e.g., XGBoost) is among the most powerful classifiers for tabular data and is widely used in particle physics for event selection.

## The Fisher Linear Discriminant

The **Fisher discriminant** finds the single linear combination of features that best separates two classes. It connects directly to the variance decomposition from [ANOVA](./anova): just as ANOVA separates between-group from within-group variance, Fisher finds the direction that maximizes the ratio of between-class to within-class scatter.

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

For predicting continuous targets, the same algorithms adapt. Linear regression minimizes squared residuals (the [chi-square method](./chi-square-method) without the weighting by uncertainties). **Ridge** and **lasso** regression add penalty terms:

$$
\hat{\mathbf{w}} = \arg\min_{\mathbf{w}} \sum_i (y_i - \mathbf{w}^T\mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_p^p.
$$

These penalties serve the same purpose as the nuisance parameter constraints in [advanced fitting](./advanced-fitting-calibration) — they prevent the model from over-adapting to noise:

* **Lasso** ($p=1$) performs automatic feature selection by driving irrelevant coefficients to exactly zero. Useful when you suspect many features are unimportant.
* **Ridge** ($p=2$) shrinks all coefficients toward zero without eliminating any. Better when features are correlated.

The penalty $\lambda \|\mathbf{w}\|^2$ is mathematically identical to placing a Gaussian prior on $\mathbf{w}$ and doing MAP estimation ([Bayesian statistics](./bayesian-statistics)). Regularization *is* Bayesian inference. The "regularization strength" $\lambda$ is the prior's precision. This isn't just an analogy — it's an exact mathematical equivalence.

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

**Principal component analysis** (PCA) finds the directions of maximum variance in the data. The principal components are the eigenvectors of the covariance matrix (which you first encountered in [introduction and concepts](./introduction-concepts)). Projecting onto the top $d$ components reduces dimensionality while preserving the most information — useful for visualization and for removing noise before applying other methods.

**Factor analysis** identifies latent (hidden) factors that explain correlations among observed variables. Unlike PCA, which simply finds directions of maximum variance, factor analysis posits a generative model: observations are linear combinations of a smaller number of unobserved factors plus noise. It answers the question: what underlying causes could produce the correlations we see?

## Model Evaluation

The critical concern in ML is **generalization**: how well does the model perform on data it has never seen? A model that memorizes the training set perfectly but fails on new data is useless — it has learned the noise, not the signal.

**Cross-validation** provides a practical estimate of generalization error. Split the data into $k$ folds, train on $k-1$, and test on the held-out fold, rotating through all folds. The average test performance estimates how the model will perform in the wild. This is the model comparison idea from [advanced fitting](./advanced-fitting-calibration), made practical.

For classifiers, key metrics include:

* **Accuracy**: Fraction of correct predictions. Simple but misleading when classes are imbalanced.
* **Precision**: $\text{TP} / (\text{TP} + \text{FP})$ — of the events you selected, how many were actually signal? This is the purity of your selection.
* **Recall (sensitivity)**: $\text{TP} / (\text{TP} + \text{FN})$ — of all the signal events that existed, how many did you find? This is the efficiency.
* **ROC curve**: Plots true positive rate vs. false positive rate as the decision threshold varies. The area under the curve (AUC) summarizes discrimination power in a single number.

**Overfitting** occurs when the model memorizes training noise rather than learning the underlying pattern. It's detected by a gap between training and validation performance. Mitigation strategies include regularization (penalty terms, as in ridge/lasso), early stopping (halting training before over-adaptation), and ensemble methods (averaging over multiple models).

> **Challenge.** Explain overfitting to a friend. Use this analogy: memorizing the textbook word-for-word lets you ace the practice test, but if the exam has different questions, you fail. A student who *understands* the material can handle both. One minute.

## Big Ideas

* Machine learning is not a new field — it is statistics, rebranded and turbocharged by computation. Recognizing the statistical ideas inside the algorithms lets you reason about when they will work and when they will fail.
* Regularization and Bayesian priors are the same operation. A ridge penalty on model weights is a Gaussian prior; lasso is a Laplace prior. The "regularization strength" is the prior's precision.
* Cross-validation is the empirical implementation of the model selection idea from advanced fitting: hold out data, measure performance on the held-out set, and use that as your honest estimate of generalization error.
* Overfitting is not a flaw of ML — it is a fundamental tension between fitting the data you have and generalizing to data you haven't seen. Every model selection decision is a navigation of this trade-off.

## What Comes Next

You have now traveled the full arc of applied statistics: summarizing data, modeling it with distributions, propagating uncertainties, fitting models, testing hypotheses, comparing groups, designing experiments, handling hierarchical and longitudinal structure, reasoning Bayesian-style, managing systematics in complex fits, and connecting all of it to machine learning.

These tools do not exist in isolation — each lesson built on the previous ones, and the connections run in both directions. The chi-square statistic is maximum likelihood in disguise. ANOVA is a special case of regression. Regularization is a Bayesian prior. The deepest skill this topic teaches is not any individual method, but the habit of asking: what is the underlying probabilistic structure of my problem, and which tool is honestly suited to answer my question given the data I actually have?

## Check Your Understanding

1. A neural network achieves 99% accuracy on the training set and 72% on the validation set. A decision tree achieves 91% on training and 89% on validation. Which model is preferable for deployment, and what does the gap between training and validation accuracy tell you in each case?
2. Ridge regression and lasso regression both add a penalty term to the loss function. In what sense is ridge regression placing a Gaussian prior on the coefficients, and what prior corresponds to lasso? Why does lasso produce sparse solutions while ridge does not?
3. You apply PCA to a dataset with 50 features and find that the first two principal components explain 85% of the variance. A colleague suggests you should always retain enough components to explain at least 95% of the variance. What is wrong with this as a universal rule, and what should you consider instead?

## Challenge

You are classifying particle physics events as signal or background using a boosted decision tree trained on simulated data. After deployment on real data, the model's performance degrades. Systematically diagnose what might have gone wrong: consider the possibility that (a) the simulation does not perfectly match the real data distribution, (b) the model overfit to simulation-specific artifacts, and (c) the feature distributions in real data have shifted over time. For each possibility, describe a diagnostic test you could run, and propose a mitigation strategy. How would the uncertainty quantification tools from this topic — confidence intervals, systematic uncertainties, calibration — help you detect and correct the problem?
