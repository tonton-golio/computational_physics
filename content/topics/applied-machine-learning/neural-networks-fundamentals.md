# Neural Network Fundamentals

Neural networks are parameterized function approximators that learn representations directly from data. Instead of hand-engineering features, you give the network raw input and let it discover the patterns. This page introduces the building blocks you need before studying specialized architectures like RNNs and GNNs.

## The multilayer perceptron

A feedforward network (MLP) with $L$ layers computes:

$$
\mathbf{h}^{(l)}=\sigma\bigl(W^{(l)}\mathbf{h}^{(l-1)}+\mathbf{b}^{(l)}\bigr),\quad l=1,\ldots,L
$$

In words: each layer takes the previous layer's output, applies a linear transformation (multiply by weights $W$, add bias $b$), then passes the result through a nonlinear activation function $\sigma$. The input is $\mathbf{h}^{(0)}=\mathbf{x}$, and the final layer produces predictions — logits for classification, continuous values for regression.

Why does stacking layers help? Each layer composes a new set of features from the previous ones. The first layer might detect edges; the second, shapes; the third, objects. Depth lets the network build increasingly abstract representations.

## Activation functions

Without a nonlinear activation, stacking layers would be pointless — a chain of linear transformations is just one big linear transformation. The activation function is what gives neural networks their power.

**ReLU** ($\max(0,x)$) is the workhorse. It is simple, fast, and its gradient is either 0 or 1, which helps with training. The downside: if a neuron's input is always negative, its output is permanently zero — a "dead neuron" that never recovers.

**Leaky ReLU** ($\max(\alpha x, x)$ with small $\alpha>0$) fixes the dead neuron problem by giving a small slope to negative inputs.

**Sigmoid** ($1/(1+e^{-x})$) squashes output to $(0,1)$, making it natural for probability outputs. But for large $|x|$, the gradient nearly vanishes, making deep networks hard to train.

**Tanh** ($\tanh(x)$) is a zero-centered version of sigmoid with the same vanishing gradient issue.

**GELU / SiLU** are smooth approximations to ReLU used in modern architectures like transformers. They give slightly better training dynamics in practice.

Start with ReLU for hidden layers unless you have a specific reason to choose otherwise.

## Backpropagation

Here is the key question: the network has thousands (or millions) of parameters, and the loss is a single number. How do we figure out which parameters to adjust, and by how much?

Imagine you are a principal grading a school's exam results. The students did poorly, but you do not know if the problem was the math teacher, the English teacher, or the P.E. coach. So you trace the blame backward: the final grades depend on the teachers' contributions, which depend on the curriculum, which depends on the department heads. You pass the blame backward through the chain until you know exactly how much each person contributed to the failure.

That is backpropagation. Using the chain rule of calculus, we propagate error gradients from the output layer back through every layer to every parameter:

$$
\frac{\partial\mathcal{L}}{\partial W^{(1)}}=\frac{\partial\mathcal{L}}{\partial\mathbf{h}^{(2)}}\cdot\frac{\partial\mathbf{h}^{(2)}}{\partial\mathbf{h}^{(1)}}\cdot\frac{\partial\mathbf{h}^{(1)}}{\partial W^{(1)}}
$$

Each term in this chain tells us: "How much does a small change in this layer's output affect the next?" Multiplied together, they give the gradient of the loss with respect to the first layer's weights. Modern frameworks (PyTorch, JAX) compute these gradients automatically, but understanding the chain rule structure still matters for diagnosing training failures — when gradients explode or vanish, it is usually because one of these terms is too large or too small.

## Regularization

Neural networks have many parameters, and given enough capacity they will happily memorize the training data — including the noise. Regularization techniques fight this.

**Dropout** randomly zeroes out a fraction of activations during training. This forces the network to learn redundant representations — no single neuron can become a critical bottleneck. It acts as an implicit ensemble: each training step uses a different random sub-network.

**Weight decay** ($L_2$ regularization) adds $\lambda\|\theta\|_2^2$ to the loss, penalizing large weights. This encourages smoother functions that are less likely to memorize noise.

**Batch normalization** normalizes activations within each mini-batch to zero mean and unit variance. This stabilizes training and allows higher learning rates.

**Early stopping** monitors validation loss and halts training when it starts increasing. It is the simplest regularization and often the most effective — your model is telling you it has learned all the real patterns and is now memorizing noise.

## Universal approximation

A remarkable theorem (Cybenko, 1989) says that a single hidden layer with enough neurons can approximate any continuous function on a compact set. So why use deep networks? Because a single wide layer can approximate anything in theory, but it may need an astronomically large number of neurons. Depth is a Swiss Army knife — deep networks with fewer neurons per layer learn more efficiently, because each layer composes features hierarchically. Width is a sledgehammer; depth is elegance.

## Architecture landscape

Neural networks specialize by changing how layers are connected. Here is the family portrait:

| Architecture | Structure | Best for |
|---|---|---|
| MLP | Fully connected layers — every neuron talks to every neuron | Tabular data, baselines |
| CNN | Convolutional filters that slide over the input — looking through a moving window | Images, spatial data |
| RNN / LSTM | Recurrent connections that carry a hidden state through time | Sequences, time series |
| GNN | Message passing between neighbors on a graph | Relational / graph data |
| Transformer | Self-attention mechanism — every token looks at every other token | Language, long-range dependencies |
| Autoencoder | Encoder-decoder bottleneck for compression | Compression, generation |

The remaining pages in this module cover RNNs and GNNs. For CNNs, transformers, and autoencoders, see [Advanced Deep Learning](/topics/advanced-deep-learning).

## Practical checklist

- Start with a small network and increase capacity if underfitting.
- Use ReLU activations and Adam optimizer as defaults.
- Apply dropout (0.1–0.5) and monitor train vs. validation loss.
- Normalize inputs and consider batch normalization for deeper networks.
- Visualize learned features and gradients to diagnose training issues.

## Check your understanding

- Can you explain backpropagation to a non-technical friend using the "passing blame backward" analogy?
- What is the one picture in your head that captures why depth is better than width?
- What experiment would you run to determine if your network needs more capacity or more regularization?
