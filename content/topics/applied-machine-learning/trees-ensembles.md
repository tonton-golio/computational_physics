# Decision Trees and Ensemble Methods

Have you ever played twenty questions? You think of an animal, and your friend asks yes-or-no questions to narrow it down. "Does it have fur?" "Is it bigger than a cat?" "Does it live in water?" Each question splits the remaining possibilities into two groups. That is exactly what a decision tree does — except it asks the questions that split the data *best*.

## Decision trees

A decision tree recursively partitions feature space by choosing, at each node, the question (feature and threshold) that produces the purest child groups. But how do we measure "pure"?

**Gini impurity** asks: if you picked two random samples from this group, what is the probability they belong to different classes?

$$
G(S)=1-\sum_{k=1}^{K}p_k^2
$$

When every sample in a group belongs to the same class, $G=0$ (perfectly pure). When classes are evenly mixed, $G$ is at its maximum. Gini is fast to compute and works well in practice.

**Entropy** comes from information theory and asks a different question: how surprised are you by a random sample from this group?

$$
H(S)=-\sum_{k=1}^{K}p_k\log p_k
$$

If every sample is class A, you are never surprised — entropy is zero. If the group is a 50/50 mix, every sample is maximally surprising — entropy is at its peak. In practice, Gini and entropy give very similar trees. Gini is the default in most libraries because it avoids the logarithm.

Trees are wonderfully interpretable — you can draw them on a whiteboard and explain every prediction. But a single deep tree will memorize the training data, fitting noise along with signal. That is where ensembles come in.

## Ensemble methods

The key insight behind ensembles is that combining many imperfect models can produce something much better than any individual.

**Random forests (bagging)** take the "wisdom of crowds" approach. Imagine 100 people each given a slightly different, incomplete map of a city. Individually, each person will get lost sometimes. But if you ask all 100 for directions and go with the majority vote, you almost never get lost. Random forests train many trees, each on a random subset of the data and a random subset of features. The trees make independent errors, and averaging cancels those errors out. Bagging reduces variance — the predictions become more stable without significantly increasing bias.

**Gradient boosting (AdaBoost, XGBoost, LightGBM)** takes the opposite approach. Instead of training many trees in parallel, it trains them sequentially. Each new tree focuses specifically on the mistakes the previous trees made. Think of it as one very determined hiker who, after every wrong turn, studies exactly where they went wrong and adjusts their strategy. Boosting reduces bias — the model becomes progressively better at capturing complex patterns, though you need to be careful not to overfit by boosting too many rounds.

## Bias-variance intuition

Here is the core tradeoff in all of machine learning, and trees make it beautifully concrete.

A very shallow tree (say, depth 2) is like a tourist with a simple rule: "If you are north of the river, go east." It is too simple to capture the real layout of the city — that is **high bias**, underfitting. A very deep tree that memorizes every turn you have ever taken is the opposite problem: it works perfectly on streets you have seen but fails on new ones — that is **high variance**, overfitting.

Bagging (random forests) fights variance by averaging many high-variance trees — each tree overfits differently, and the errors cancel. Boosting fights bias by building a sequence of simple trees that progressively correct each other's mistakes. In practice, gradient boosting with careful tuning (early stopping, regularization) is one of the most powerful methods for structured data.

[[figure aml-xor-ensemble]]

## Interactive simulations

[[simulation aml-tree-split-impurity]]

[[simulation aml-tree-ensemble-xor]]

## Model selection notes

- Always use cross-validation for small tabular datasets — a single train/test split is too noisy.
- Key hyperparameters to tune: `max_depth`, `min_samples_leaf`, `n_estimators`, and learning rate (for boosting).
- When you use predicted probabilities for decisions (not just class labels), run calibration checks — tree ensembles can produce poorly calibrated probabilities.

## Check your understanding

- Can you explain to a non-technical friend what a decision tree does using the twenty-questions game?
- What single mental snapshot captures why averaging many noisy predictions reduces error?
- What experiment would reveal whether your tree ensemble is overfitting?
