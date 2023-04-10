
KEY: intro
To get started with artificial neural networks, we'll work with the MNIST dataset. This dataset contains images (28x28) of handwritten digits, each of which is labeled with the digit it represents.


KEY: multilayer perceptron model
We load data with
```python
train_data = datasets.MNIST(root=filepath_assets+'data', 
                            train=True, download=True, 
                            transform=transforms.ToTensor())
```

We then pass it into our data loader, which makes it ready for the model to digest.
```python
train_loader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
```


The multilayer perceptron model is a feedforward neural network. It consists of an input layer, one or more hidden layers, and an output layer.
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

We then instanciate our model, define our criterion and optimizer;
```python
model = MultilayerPerceptron()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```


KEY: training
We then train our model. Below is a visualization of the process, with loss and accuracy being displayed.


KEY: evaluation

