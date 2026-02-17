# Artificial Neural Networks

## Introduction

To get started with artificial neural networks, we work with the **MNIST dataset**. This dataset contains 28x28 grayscale images of handwritten digits (0-9), each labeled with the digit it represents.

[[simulation adl-activation-functions]]

## Multilayer Perceptron Model

We load data with:
```python
train_data = datasets.MNIST(root=filepath_assets+'data',
                            train=True, download=True,
                            transform=transforms.ToTensor())
```

We then pass it into a data loader, which batches and shuffles the data for training:
```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
```

The **multilayer perceptron** (MLP) is a feedforward neural network consisting of an input layer, one or more hidden layers, and an output layer:
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

We then instantiate the model, define the loss function and optimizer:
```python
model = MultilayerPerceptron()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

## Training and Evaluation

We train the model by iterating over batches, computing the loss, and updating parameters via backpropagation. The training loop tracks loss and accuracy over epochs.
