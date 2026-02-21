# Artificial Neural Networks

## The simplest possible learner

Before we build anything complicated, let us train the simplest thing that can learn. A single artificial neuron takes a handful of numbers as input, multiplies each one by a weight, adds them up, and then asks: "Is the total big enough?" That is all a neuron does. It is a tiny voting machine: each input casts a weighted vote, and the neuron makes a decision based on the tally.

Mathematically, the neuron computes a weighted sum of inputs plus a bias, then applies a nonlinear **activation function**:

$$
y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b).
$$

The weights $w_i$ determine how much each input contributes, the bias $b$ shifts the decision boundary, and $\sigma$ introduces nonlinearity. This abstraction is loosely inspired by biological neurons: dendrites receive signals, the cell body integrates them, and the axon fires when a threshold is exceeded.

## The perceptron

The simplest neural network is a single neuron called the **perceptron**. For binary classification, the perceptron computes:

$$
\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w}^T \mathbf{x} + b > 0, \\ 0 & \text{otherwise}. \end{cases}
$$

This defines a linear decision boundary (a hyperplane) in the input space. When the perceptron gets a prediction wrong, it adjusts its weights in the direction of the mistake:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \, (y - \hat{y}) \, \mathbf{x},
$$

where $\eta$ is the learning rate. The perceptron converges for linearly separable data but cannot solve nonlinear problems like XOR. This limitation motivated the development of multilayer networks.

## Your first training loop

Before we go any further, let us actually train something. Here is the simplest possible neural network on the MNIST handwritten digit dataset. We load the data, build a model, and train it in under 20 lines. Run this, watch the loss go down, and then we will explain every piece:

```python
train_data = datasets.MNIST(root='data', train=True, download=True,
                            transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)

model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
<!--code-toggle-->
```pseudocode
train_data = LOAD_DATASET("MNIST", split="train")
APPLY_TRANSFORM(train_data, to_tensor)
train_loader = CREATE_DATALOADER(train_data, batch_size=100, shuffle=True)

model = SEQUENTIAL(FLATTEN, LINEAR(784, 10))
criterion = CROSS_ENTROPY_LOSS()
optimizer = ADAM(model.parameters(), lr=0.001)

FOR epoch = 1 TO 5:
    FOR EACH (batch_x, batch_y) IN train_loader:
        ZERO_GRADIENTS(optimizer)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        BACKWARD(loss)
        UPDATE_PARAMETERS(optimizer)
    PRINT("Epoch", epoch, "Loss:", loss)
```

That single linear layer already gets about 92% accuracy on digit recognition. But it can only draw straight decision boundaries. To do better, we need nonlinearity and depth.

## Activation functions

Why do we need the activation function $\sigma$ at all? Without it, stacking linear layers would still produce a linear function — no matter how many layers you add, the composition of linear functions is still linear. The activation function is what gives the network the ability to bend and curve its decision boundaries.

Common choices include:

* **ReLU**: $f(x) = \max(0, x)$. Dead simple: if the input is positive, pass it through; if negative, output zero. Fast and effective, but neurons can "die" (output zero for all inputs) if the bias drifts too negative.
* **Sigmoid**: $f(x) = 1 / (1 + e^{-x})$. Squashes everything into $(0, 1)$, useful for output layers in binary classification. But it has a fatal flaw for deep networks: the derivative $f'(x) = f(x)(1 - f(x))$ is at most $0.25$, so gradients shrink exponentially through many layers. This is the **vanishing gradient problem**.
* **Tanh**: $f(x) = \tanh(x)$. Outputs in $(-1, 1)$, zero-centered. Better gradient flow than sigmoid but still saturates for large $|x|$.
* **Swish**: $f(x) = x \cdot \sigma(x)$. Smooth and non-monotonic. Used in EfficientNet and modern architectures; often outperforms ReLU in deep networks.

ReLU and its variants (Leaky ReLU, GELU) dominate modern practice because they maintain strong gradients even in deep networks.

[[simulation adl-activation-functions]]

## What if we didn't have activation functions?

Without nonlinear activations, a 100-layer network would compute exactly the same thing as a single-layer network. The entire stack of matrix multiplications would collapse into one giant matrix multiplication. You could never learn anything more complex than a straight line through the data. Activations are the ingredient that turns a linear calculator into a universal function approximator.

## Multilayer perceptrons

A **multilayer perceptron** (MLP) stacks multiple layers of neurons. The **universal approximation theorem** says something remarkable: a feedforward network with a single hidden layer containing sufficiently many neurons can approximate any continuous function on a compact set to arbitrary accuracy. The network does not need to be deep — it just needs to be wide enough.

So why bother with depth? In practice, **deeper networks** (more layers, fewer neurons per layer) are vastly more efficient:
* Depth enables hierarchical feature extraction: early layers detect simple patterns, later layers compose them into complex concepts.
* Certain functions require exponentially many neurons in a shallow network but only polynomially many in a deep one. Depth gives you exponential compression.

## Forward pass

The MLP processes an input by passing it through each layer sequentially. For an input $\mathbf{x} \in \mathbb{R}^{784}$ (a flattened 28x28 image), imagine the data flowing through a pipeline: each layer transforms the representation, extracts more abstract features, and passes the result to the next layer. A two-hidden-layer MLP computes:

$$
\mathbf{h}_1 = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1), \qquad \mathbf{h}_2 = \text{ReLU}(W_2 \mathbf{h}_1 + \mathbf{b}_2), \qquad \hat{\mathbf{y}} = \text{softmax}(W_3 \mathbf{h}_2 + \mathbf{b}_3).
$$

At each layer the dimensions change: $784 \to 120 \to 84 \to 10$. The final softmax converts logits into a probability distribution over the 10 digit classes.

```python
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120, 84]):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], out_sz)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return F.log_softmax(x, dim=1)
```
<!--code-toggle-->
```pseudocode
CLASS MultilayerPerceptron:
    INIT(in_sz=784, out_sz=10, layers=[120, 84]):
        fc1 = LINEAR(in_sz, layers[0])
        fc2 = LINEAR(layers[0], layers[1])
        out = LINEAR(layers[1], out_sz)

    FORWARD(x):
        x = RELU(fc1(x))
        x = RELU(fc2(x))
        x = out(x)
        RETURN LOG_SOFTMAX(x, dim=1)
```

## Loss functions

How do we tell the network it made a mistake? We need a single number that measures how wrong the predictions are. That number is the **loss function**.

* **Cross-entropy loss**: The standard choice for classification. For a true label $y$ and predicted probabilities $\hat{p}$:

$$
\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{p}_c.
$$

For one-hot encoded labels, this simplifies to $\mathcal{L} = -\log \hat{p}_{y}$ where $y$ is the correct class. The intuition: if the model is 99% confident in the right answer, the loss is tiny ($-\log 0.99 \approx 0.01$). If it is only 1% confident, the loss is enormous ($-\log 0.01 \approx 4.6$). Cross-entropy punishes confident wrong answers severely.

* **Mean squared error** (MSE): Used for regression tasks:

$$
\mathcal{L}_{\text{MSE}} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2.
$$

MSE is not ideal for classification because its gradients become very small when the prediction is confident but wrong — exactly when you need the strongest learning signal.

## Backpropagation

**Backpropagation** is how the network learns from its mistakes. Imagine a game of telephone: the output layer says "I got it wrong by this much," and that error message is passed backward through the network, one layer at a time. Each layer adjusts its weights based on the portion of the error it was responsible for.

Formally, backpropagation applies the **chain rule** systematically through the computational graph. For a single weight $w_{jk}^{(l)}$ in layer $l$, the gradient is:

$$
\frac{\partial \mathcal{L}}{\partial w_{jk}^{(l)}} = \frac{\partial \mathcal{L}}{\partial a_j^{(l)}} \cdot \frac{\partial a_j^{(l)}}{\partial w_{jk}^{(l)}} = \delta_j^{(l)} \cdot h_k^{(l-1)},
$$

where $\delta_j^{(l)} = \frac{\partial \mathcal{L}}{\partial a_j^{(l)}}$ is the "error signal" at neuron $j$ in layer $l$, and $h_k^{(l-1)}$ is the activation from the previous layer. The key insight is that $\delta$ propagates backward:

$$
\delta_j^{(l)} = \sigma'(a_j^{(l)}) \sum_i w_{ij}^{(l+1)} \delta_i^{(l+1)}.
$$

This recursion means we compute all gradients in a single backward pass, making training efficient even for networks with millions of parameters. Without backpropagation, you would have to wiggle each weight individually and measure the change in loss — millions of forward passes instead of one backward pass.

[[simulation adl-backprop-flow]]

## What if we didn't have backpropagation?

Without backpropagation, training a network with a million parameters would require at least a million forward passes per update step (one per parameter) to estimate each gradient numerically. A single training step that takes milliseconds with backpropagation would take hours. Modern deep learning would simply not exist.

## The full training loop

Now let us put all the pieces together. Each training step performs four operations: (1) forward pass to compute predictions, (2) loss computation, (3) backward pass to compute gradients, (4) parameter update via the optimizer:

```python
model = MultilayerPerceptron()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        output = model(batch_x.view(-1, 784))
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
```
<!--code-toggle-->
```pseudocode
model = MultilayerPerceptron()
criterion = CROSS_ENTROPY_LOSS()
optimizer = ADAM(model.parameters(), lr=0.001)

FOR epoch = 1 TO 10:
    FOR EACH (batch_x, batch_y) IN train_loader:
        ZERO_GRADIENTS(optimizer)
        output = model(FLATTEN(batch_x))
        loss = criterion(output, batch_y)
        BACKWARD(loss)
        UPDATE_PARAMETERS(optimizer)
```

Training iterates over the dataset in **epochs** (one full pass through all training data), processing **mini-batches** at each step. **Convergence** is monitored by tracking loss and validation accuracy across epochs. When the validation loss stops decreasing while the training loss keeps falling, the model is overfitting — a signal to stop training or add regularization.

## Big Ideas

* Without nonlinearity, a million-layer network is still just a straight line through the data — activations are the ingredient that makes depth matter.
* Backpropagation is the chain rule applied cleverly: one backward pass computes every gradient simultaneously, turning what would take months into milliseconds.
* The universal approximation theorem promises that a wide enough single hidden layer can fit anything, but depth is what makes that fitting *efficient*.
* A neuron that outputs zero for all inputs contributes nothing — the vanishing gradient and dying ReLU problems are the same story told by different characters.

## What Comes Next

You now have a network that can learn, but you have left enormous questions on the table: what learning rate should you use, how do you stop the network from memorizing noise, and how do you find good hyperparameters in a space with hundreds of dimensions? The next lesson takes on the optimizer — the thing that actually moves the weights — and the full toolkit of regularization, learning rate schedules, and initialization strategies that separate a network that trains from one that generalizes.

The transition from "this works in principle" to "this works in practice" is where most of the craft lives, and understanding the optimizer landscape will change how you think about every network you train from here on.

## Check Your Understanding

1. A single neuron with no activation function can only draw a straight-line decision boundary. What happens when you stack ten such neurons in sequence — does the boundary become more complex? Why or why not?
2. The perceptron update rule moves the weights toward the correct class when the prediction is wrong. Why does this rule fail on the XOR problem even after many iterations?
3. Backpropagation computes the gradient of the loss with respect to every weight in one backward pass. Why is this so much cheaper than perturbing each weight individually to estimate gradients numerically?

## Challenge

Design a training experiment to test whether depth or width matters more for learning a target function. Pick a specific target — say, $f(x) = \sin(10x)$ on $[0, 1]$ — and compare networks of equal parameter count but different shapes (one wide hidden layer vs. many narrow layers). Plot the approximation error as a function of training time and total parameters. Does the answer depend on the target function? What happens if you replace $\sin(10x)$ with a function that has a natural hierarchical structure, like a composition of simpler functions?

