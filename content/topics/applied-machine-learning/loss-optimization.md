# Loss Functions and Optimization

Imagine a blindfolded hiker in the mountains who can only feel the slope under their feet. Every step they take downhill is gradient descent. The steeper the slope, the bigger the step -- until they reach the bottom (a minimum of the loss). Everything in machine learning starts here: you pick a number that measures how wrong your predictions are (the loss), and then you adjust your model's parameters to make that number smaller.

## Classification losses

When the task is to assign labels -- spam or not spam, cat or dog -- we need a loss that punishes wrong labels.

**Zero-one loss** is the most natural: count the mistakes. But it is flat everywhere except at the decision boundary, which means it gives the optimizer no slope to follow. You cannot walk downhill on a plateau.

**Hinge loss** fixes this by enforcing a margin. "Not only should you get the label right, you should be *confident* about it." This is the loss behind support vector machines.

**Binary cross-entropy** takes a probabilistic view. It says: "Tell me the probability you assign to the correct class, and I will punish you logarithmically for being wrong." Predict 0.99 for the right class? Tiny penalty. Predict 0.01? Enormous.

$$
\mathcal{L}_{BCE}=-\frac{1}{N}\sum_{i=1}^{N}\left(y_i\log\hat{p}_i+(1-y_i)\log(1-\hat{p}_i)\right)
$$

For each sample we take the log of the predicted probability for the true class and average across the dataset. Confident correct predictions cost nearly nothing; confident wrong ones blow up.

## Regression losses

When you are predicting continuous values -- house prices, temperatures, molecular energies -- you need a different penalty.

Think of it this way. Suppose you are estimating earthquake damage across a city. **MSE (mean squared error)** squares every error before averaging. A house where you are off by \$100k gets 100 times the penalty of one where you are off by \$10k. MSE *hates* large errors and will bend over backwards to fix them, even at the cost of making small errors slightly worse.

**MAE (mean absolute error)** treats all errors proportionally. Off by \$10k? Penalty of \$10k. Off by \$100k? Penalty of \$100k. It does not obsess over outliers. But it has a kink at zero, which makes optimization trickier.

**Huber loss** gives you the best of both worlds -- MSE for small errors (smooth, easy to optimize) and MAE for large errors (robust to outliers).

[[figure aml-loss-outlier-comparison]]

## Optimization dynamics

Back to our blindfolded hiker. Gradient descent updates every parameter by taking a step proportional to the downhill slope:

$$
\theta_{t+1}=\theta_t-\eta \nabla_{\theta}\mathcal{L}(\theta_t)
$$

Here $\eta$ is the learning rate -- your step size. This single number controls everything. Too large and you overshoot the valley and bounce around wildly, possibly diverging. Too small and you inch forward painfully, stuck in some shallow dip that is not the true minimum.

Modern optimizers improve on vanilla gradient descent. **Momentum** remembers past steps and keeps rolling in a consistent direction, like a ball with inertia. **RMSProp** adapts the step size per parameter -- bigger steps in flat directions, smaller ones in steep ones. **Adam** combines both and is the default starting point for most practitioners.

## Validation and overfitting

Here is the most important discipline in all of machine learning: always separate your data into three pools.

Your **training set** is where the model learns -- it adjusts parameters by minimizing the loss on these samples. Your **validation set** is your mirror -- you check performance here after each round of training, but you never train on it. Your **test set** is sacred. You touch it once, at the very end, to get an honest estimate of real-world performance.

Watch the training and validation loss curves together. Early on, both go down -- the model is learning real patterns. At some point the training loss keeps falling but the validation loss starts creeping up. That is the moment overfitting begins: the model is memorizing noise instead of learning structure. The best parameters correspond to the lowest validation loss.

When data is limited, k-fold cross-validation lets every sample take a turn as the validator -- average the scores and you get an honest generalization estimate. For time series, you never let the future leak into the past -- that is cheating.

[[simulation aml-loss-landscape]]

## So what did we really learn?

The loss function is not an arbitrary scorecard -- it encodes your assumptions about what kinds of errors hurt most. MSE hates outliers; MAE forgives them; Huber compromises. Choosing the wrong loss means optimizing for the wrong thing.

Gradient descent is remarkably universal: the same blindfolded hiker algorithm, with minor variations, trains a logistic regression, a 100-layer neural network, and everything in between.

And overfitting is not a failure of the model -- it is a message. The model learned something real, then kept going and started learning noise. Your job is to know when to stop listening.

## Challenge

Your colleague insists on using a single 80/20 train-test split instead of k-fold cross-validation, arguing it is "good enough." Design an experiment using a small synthetic dataset that either vindicates or refutes their claim. Generate data with a known ground truth, compare the single-split estimate to the k-fold estimate, and measure how often each one is closer to the true error. Under what conditions does the single split fail most dramatically?
