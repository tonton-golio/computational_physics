# Decision Trees and Ensemble Methods

Have you ever played twenty questions? You think of an animal, and your friend asks yes-or-no questions to narrow it down. "Does it have fur?" "Is it bigger than a cat?" "Does it live in water?" Each question splits the remaining possibilities into two groups. That is exactly what a decision tree does -- except it asks the questions that split the data *best*.

## Decision trees

A decision tree recursively partitions feature space by choosing, at each node, the question (feature and threshold) that produces the purest child groups. But how do we measure "pure"?

Gini asks "how mixed is this crowd?" Entropy asks "how surprised am I by the next person I meet?" Both work; Gini is faster, so it is the default in most libraries.

$$
G(S)=1-\sum_{k=1}^{K}p_k^2 \qquad H(S)=-\sum_{k=1}^{K}p_k\log p_k
$$

When every sample in a group belongs to the same class, both scores hit zero -- perfectly pure. When classes are evenly mixed, both peak. In practice, they give very similar trees.

Trees are wonderfully interpretable -- you can draw them on a whiteboard and explain every prediction. But a single deep tree will memorize the training data, fitting noise along with signal. That is where ensembles come in.

[[figure aml-tree-growth-steps]]

## Ensemble methods

The key insight: combining many imperfect models can produce something much better than any individual.

**Random forests (bagging)** take the "wisdom of crowds" approach. Imagine 100 people each given a slightly different, incomplete map of a city. Individually, each person gets lost sometimes. But ask all 100 for directions and go with the majority vote? You almost never get lost. Random forests train many trees, each on a random subset of the data and a random subset of features. The trees make independent errors, and averaging cancels those errors out. Bagging reduces variance -- predictions become more stable without significantly increasing bias.

**Gradient boosting (AdaBoost, XGBoost, LightGBM)** takes the opposite approach. Instead of training trees in parallel, it trains them sequentially. Each new tree focuses specifically on the mistakes the previous trees made. Think of one very determined hiker who, after every wrong turn, studies exactly where they went wrong and adjusts their strategy. Boosting reduces bias -- the model becomes progressively better at capturing complex patterns, though you need to be careful not to overfit by boosting too many rounds.

## Bias-variance intuition

Here is the core tradeoff in all of machine learning, and trees make it beautifully concrete.

A very shallow tree (depth 2) is like a tourist with one simple rule: "If you are north of the river, go east." Too simple to capture the real layout -- that is **high bias**, underfitting. A very deep tree that memorizes every turn you have ever taken works perfectly on streets you have seen but fails on new ones -- **high variance**, overfitting.

Bagging fights variance by averaging many high-variance trees. Each tree overfits differently, and the errors cancel. Boosting fights bias by building a sequence of simple trees that progressively correct each other's mistakes. In practice, gradient boosting with careful tuning (early stopping, regularization) is one of the most powerful methods for structured data.

[[simulation aml-tree-split-impurity]]

## So what did we really learn?

A decision tree is the most interpretable nonlinear model that exists -- you can literally read it out loud and explain every prediction. That interpretability is also its weakness: a single tree deep enough to be accurate will memorize the training data.

Bagging and boosting are two philosophies for turning weak models into strong ones. Bagging parallelizes imperfect estimators and averages away their independent errors. Boosting serializes them and forces each new one to focus on what the others got wrong.

And here is the gorgeous part: on structured tabular data, gradient-boosted trees *still* routinely outperform deep neural networks. Do not reach for deep learning until you have tried XGBoost.

## Challenge

Construct a minimal synthetic dataset in two dimensions where a single decision tree with unlimited depth achieves near-perfect training accuracy but poor test accuracy, while a random forest achieves both. Then flip the experiment: find a dataset where boosting outperforms the random forest. What property of the data determines which ensemble approach wins?
