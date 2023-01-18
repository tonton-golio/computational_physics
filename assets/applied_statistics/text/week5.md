

KEY: description
Week 5 (Bayesian statistics and Multivariate Analysis):
* Dec 19: Bayes theorem and Baysian statistics.
* Dec 20: Multi-Variate Analysis (MVA). Fits in 2D. The linear Fisher discriminant.

KEY: Bayes theorem and Baysian statistics
#### Bayes theorem and Baysian statistics
Bayesian statistics is a framework for statistical inference in which one's beliefs about the probability of a certain event occurring are updated as new data is obtained. This is in contrast to classical (frequentist) statistics, which assumes that the probability of an event is fixed and does not change based on the data.

**Bayes' theorem**: This is the fundamental formula in Bayesian statistics. It states that the posterior probability of a hypothesis (H) given some data (D) is proportional to the product of the prior probability of the hypothesis and the likelihood of the data given the hypothesis.
$$ 
    P(H|D) = \frac{P(D|H) \cdot P(H)}{P(D)}
$$

KEY: Multi-Variate Analysis (MVA)
#### Multi-Variate Analysis (MVA)
Multivariate analysis (MVA) is a set of statistical techniques used to analyze data that consists of more than one variable. The goal of MVA is to understand the relationships between multiple variables and how they affect one another.

There are many different techniques that fall under the umbrella of MVA, including:

* Principal component analysis (PCA): a technique that is used to identify patterns in a data set, by finding the directions of maximum variance in the data.
* Factor analysis: a technique that is used to identify underlying factors that explain the relationships between multiple variables.
* Cluster analysis: a technique that is used to group similar observations together, based on the values of multiple variables.
* Discriminant analysis: a technique that is used to classify observations into different groups, based on the values of multiple variables.
* Multivariate regression: a technique that is used to model the relationships between multiple independent variables and a single dependent variable.

KEY: The linear Fisher discriminant
#### The linear Fisher discriminant
The Fisher discriminant is based on the following formula:
$$
    w = (μ_0 - μ_1) * (S^-1)
$$
where:

$w$ is the Fisher discriminant, which is a linear combination of the features. $μ_0$ and $μ_1$ are the means of the two classes. $S$ is the pooled within-class covariance matrix.
The pooled within-class covariance matrix is a weighted average of the covariance matrices for each class, where the weights are the class priors. It is defined as:

$$
    S = (1/n) * (S_0 + S_1)
$$
where:

$S_0$ and $S_1$ are the covariance matrices for the two classes. $n$ is the total number of observations.

The value of the Fisher discriminant for a new observation is calculated using the following formula:

$$
w^T \times x
$$

where:

$w^T$ is the transpose of the Fisher discriminant.
$x$ is the new observation, which is a vector of the feature values.

It's important to note that this formula holds under certain assumptions such as the classes have normal distribution and the covariance matrices are identical, otherwise different approaches should be considered.