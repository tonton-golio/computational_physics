# Artificial Neural Networks

## The simplest possible learner

Before we build anything complicated, let's train the simplest thing that can learn. A single artificial neuron takes a handful of numbers as input, multiplies each one by a weight, adds them up, and asks: "Is the total big enough?" That's it. It's a tiny voting machine -- each input casts a weighted vote, and the neuron makes a decision based on the tally.

Mathematically, the neuron computes a weighted sum plus a bias, then applies a nonlinear **activation function**:

$$
y = \sigma\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = \sigma(\mathbf{w}^T \mathbf{x} + b).
$$

The weights $w_i$ determine how much each input contributes, the bias $b$ shifts the decision boundary, and $\sigma$ introduces nonlinearity.

## The perceptron

The simplest neural network is a single neuron called the **perceptron**. For binary classification it computes:

$$
\hat{y} = \begin{cases} 1 & \text{if } \mathbf{w}^T \mathbf{x} + b > 0, \\ 0 & \text{otherwise}. \end{cases}
$$

When it gets a prediction wrong, it adjusts its weights toward the mistake:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta \, (y - \hat{y}) \, \mathbf{x}.
$$

This converges for linearly separable data but can't solve nonlinear problems like XOR. That limitation is what forced people to invent multilayer networks.

## Your first training loop

Enough theory. Let's actually train something. Here's the simplest possible neural network on MNIST handwritten digits -- load the data, build a model, train it in under 20 lines. Run this, watch the loss go down, and then we'll explain every piece:

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

Why do we need the activation function $\sigma$ at all? Here's the thing -- without it, stacking layers does nothing. The composition of linear functions is still linear. You could pile a hundred layers deep and the whole stack would still compute one giant matrix multiplication. Activations are the ingredient that lets the network *bend*.

The main characters:

* **ReLU**: $f(x) = \max(0, x)$. Positive input passes through, negative input becomes zero. Dead simple, fast, and effective -- but neurons can "die" if the bias drifts too negative and they output zero for every input.
* **Sigmoid**: $f(x) = 1 / (1 + e^{-x})$. Squashes everything into $(0, 1)$. The fatal flaw for deep networks: its derivative maxes out at $0.25$, so gradients shrink exponentially through many layers. That's the **vanishing gradient problem**.
* **Tanh**: $f(x) = \tanh(x)$. Zero-centered, which helps, but still saturates for large inputs.
* **Swish/GELU**: Smooth, non-monotonic modern alternatives that often outperform ReLU in deep networks.

ReLU and its variants dominate modern practice because they keep gradients strong even in deep networks.

## Multilayer perceptrons

A **multilayer perceptron** (MLP) stacks multiple layers of neurons. And here's the gorgeous part -- the **universal approximation theorem** says a feedforward network with a single hidden layer containing enough neurons can approximate *any* continuous function to arbitrary accuracy.

So why bother with depth? Because in practice, deeper networks are vastly more efficient. Depth gives you hierarchical feature extraction: early layers detect simple patterns, later layers compose them into complex concepts. Certain functions need exponentially many neurons in a shallow network but only polynomially many in a deep one. Depth is exponential compression.

## Forward pass

The MLP processes input by passing it through each layer sequentially. Think of data flowing through a pipeline where each layer transforms the representation and extracts more abstract features. A two-hidden-layer MLP computes:

$$
\mathbf{h}_1 = \text{ReLU}(W_1 \mathbf{x} + \mathbf{b}_1), \qquad \mathbf{h}_2 = \text{ReLU}(W_2 \mathbf{h}_1 + \mathbf{b}_2), \qquad \hat{\mathbf{y}} = \text{softmax}(W_3 \mathbf{h}_2 + \mathbf{b}_3).
$$

Dimensions change at each layer: $784 \to 120 \to 84 \to 10$. The final softmax converts logits into a probability distribution over 10 digit classes.

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

How do we tell the network it messed up? We need a single number that measures how wrong the predictions are. That number is the **loss function**.

**Cross-entropy loss** is the standard for classification. The intuition is beautiful: if the model is 99% confident in the right answer, the loss is tiny ($-\log 0.99 \approx 0.01$). If it's only 1% confident, the loss is enormous ($-\log 0.01 \approx 4.6$). Cross-entropy *punishes confident wrong answers severely*:

$$
\mathcal{L}_{\text{CE}} = -\sum_{c=1}^{C} y_c \log \hat{p}_c.
$$

**Mean squared error** is used for regression. It's not ideal for classification because its gradients become very small when the prediction is confident but wrong -- exactly when you need the strongest learning signal.

## Backpropagation

Here's the trick that made everything possible. Imagine the error is a hot potato passed backward through a chain of friends. Each friend only needs to know how much to adjust their own throw. That's **backpropagation** -- the chain rule applied systematically through the computational graph.

For a single weight $w_{jk}^{(l)}$ in layer $l$:

$$
\frac{\partial \mathcal{L}}{\partial w_{jk}^{(l)}} = \delta_j^{(l)} \cdot h_k^{(l-1)},
$$

where the error signal $\delta$ propagates backward:

$$
\delta_j^{(l)} = \sigma'(a_j^{(l)}) \sum_i w_{ij}^{(l+1)} \delta_i^{(l+1)}.
$$

Why is this such a big deal? Without backprop, training a million-parameter network would require roughly a million separate forward passes per update -- just to estimate each gradient by wiggling one weight at a time. Backprop does the same job with **one** forward pass and **one** backward pass. It's the difference between "possible in theory" and "runs on my laptop tonight."

## The full training loop

Now let's put all the pieces together. Each training step does four things: forward pass, loss computation, backward pass, parameter update:

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

Training iterates over the dataset in **epochs**, processing **mini-batches** at each step. When validation loss stops decreasing while training loss keeps falling, the model is overfitting -- time to stop training or add regularization.

## Big Ideas

* Without nonlinearity, a million-layer network is still just a straight line -- activations are what make depth matter.
* Backprop is the chain rule applied cleverly: one backward pass computes every gradient simultaneously.
* The universal approximation theorem promises width is enough, but depth is what makes fitting *efficient*.

## What Comes Next

You now have a network that can learn, but you've left enormous questions on the table: what learning rate should you use? How do you stop the network from memorizing noise? How do you find good hyperparameters in a space with hundreds of dimensions? The next lesson takes on the optimizer and the full toolkit of regularization, schedules, and initialization strategies that separate a network that trains from one that generalizes.
