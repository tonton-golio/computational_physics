# Neural Network Fundamentals

Neural networks are parameterized function approximators that learn representations directly from data. Instead of hand-engineering features, you give the network raw input and let it discover the patterns. This page hands you the building blocks you need before studying specialized architectures.

## The multilayer perceptron

A feedforward network (MLP) with $L$ layers computes:

$$
\mathbf{h}^{(l)}=\sigma\bigl(W^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)}\bigr),\quad l=1,\ldots,L
$$

Each layer takes the previous layer's output, applies a linear transformation (multiply by weights $W$, add bias $b$), then passes the result through a nonlinear activation function $\sigma$. The input is $\mathbf{h}^{(0)}=\mathbf{x}$, and the final layer produces predictions -- logits for classification, continuous values for regression.

Why does stacking layers help? Each layer composes a new set of features from the previous ones. The first layer might detect edges; the second, shapes; the third, objects. Depth lets the network build increasingly abstract representations.

## Activation functions

Without a nonlinear activation, stacking layers would be pointless -- a chain of linear transformations is just one big linear transformation. The activation function is what gives neural networks their power.

**ReLU** ($\max(0,x)$) is the workhorse. Simple, fast, gradient is either 0 or 1. The downside: if a neuron's input is always negative, its output is permanently zero -- a "dead neuron" that never recovers.

**Leaky ReLU** ($\max(\alpha x, x)$ with small $\alpha>0$) fixes dead neurons by giving a small slope to negative inputs.

**Sigmoid** ($1/(1+e^{-x})$) squashes output to $(0,1)$, natural for probabilities. But for large $|x|$, the gradient nearly vanishes, making deep networks hard to train.

**Tanh** is a zero-centered sigmoid with the same vanishing gradient issue.

**GELU / SiLU** are smooth approximations to ReLU used in transformers. Slightly better training dynamics in practice.

Start with ReLU for hidden layers unless you have a specific reason to choose otherwise.

[[figure aml-activation-gallery]]

## Backpropagation: the telephone game of blame

Here is the key question: the network has thousands (or millions) of parameters, and the loss is a single number. How do we figure out which parameters to adjust, and by how much?

Think of it as a telephone game of blame. The loss is angry at the output and starts shouting "who did this?!" The blame passes backward through every layer, getting multiplied at each step by how sensitive that layer was. Layer 5 says "it was mostly layer 4's fault," and layer 4 says "blame layer 3," and so on until every parameter knows exactly how much it contributed to the screw-up.

$$
\frac{\partial\mathcal{L}}{\partial W^{(1)}}=\frac{\partial\mathcal{L}}{\partial\mathbf{h}^{(2)}}\cdot\frac{\partial\mathbf{h}^{(2)}}{\partial\mathbf{h}^{(1)}}\cdot\frac{\partial\mathbf{h}^{(1)}}{\partial W^{(1)}}
$$

That chain of multiplications is the chain rule in action. And it is why gradients can vanish or explode -- if one of those terms is consistently less than 1, the blame signal dies before it reaches the early layers. If it is greater than 1, the signal blows up. Modern frameworks (PyTorch, JAX) compute these gradients automatically, but understanding the chain structure still matters for diagnosing training failures.

[[figure aml-numeric-forward-backward]]

## Regularization

Neural networks have many parameters, and given enough capacity they will happily memorize the training data -- including the noise. Regularization fights this.

**Dropout** randomly zeroes out a fraction of activations during training, forcing the network to learn redundant representations. It acts as an implicit ensemble: each training step uses a different random sub-network.

**Weight decay** ($L_2$ regularization) adds $\lambda\|\theta\|_2^2$ to the loss, penalizing large weights and encouraging smoother functions.

**Batch normalization** normalizes activations within each mini-batch to zero mean and unit variance, stabilizing training and allowing higher learning rates.

**Early stopping** monitors validation loss and halts training when it starts increasing. Often the most effective regularizer -- your model is telling you it has learned all the real patterns.

## Universal approximation

A remarkable theorem says that a single hidden layer with enough neurons can approximate any continuous function on a compact set. So why use deep networks? Because a single wide layer may need an astronomically large number of neurons. Depth is a Swiss Army knife -- deep networks learn more efficiently because each layer composes features hierarchically. Width is a sledgehammer; depth is elegance.

Change the wiring and you get CNNs for pictures, RNNs for sequences, GNNs for relationships -- same Lego bricks, different shapes.

[[simulation aml-activation-gallery]]

## So what did we really learn?

Backpropagation is just the chain rule, applied systematically and automatically. The mystery dissolves once you see it as blame propagating backward through a chain of decisions.

Nonlinearity is not a detail -- it is the *entire point*. Without an activation function, you have an expensive way to do linear regression. With one, you have a universal function approximator.

Regularization and capacity are two sides of the same dial. Before you make the network bigger, ask whether regularization is already failing; before you add dropout, ask whether the model has enough capacity to fit the training data at all.

And depth buys efficiency in representation: what an exponentially wide single-layer network can represent, a polynomial-depth network can represent with far fewer parameters. That is why the field went deep.

## Challenge

Build a two-layer MLP by hand (on paper or in a few lines of NumPy) and train it on the XOR problem -- four input points, four labels, no library. Derive the backpropagation updates yourself using the chain rule, implement gradient descent by hand, and verify that the network converges. What is the minimum number of hidden neurons required? What happens if you use a linear activation instead of ReLU? Can you prove, not just observe, why the linear version fails?
