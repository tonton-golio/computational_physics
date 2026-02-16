
KEY: intro
Convolutional neural networks are better for image recognition, because they take advantage of spatial structures, and recognize larger scale patterns in the image.

KEY: kernels
We may choose from a variety of kernel:
* averaging (`np.ones(x,x)/x**2`)
* Gaussian blur
* Dilated (Atrous) Convolution

KEY: convolutional neural network model
```python

class ConvolutionalNetwork(nn.Module):
    '''
    We will use the following structure:
    - convolutional layer
    - max pooling layer
    - convolutional layer
    - max pooling layer
    - linear layer
    - linear layer

    The model should take in a 28x28 image and output a 10x1 vector of logits.
    '''
    def __init__(self, im_shape=(1, 28, 28), n_classes=10, batch_size=100):
        super().__init__()
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.n_classes = n_classes

        # define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=8*7*7, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 8*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

KEY: evaluation
We evaluate the model in the same way again, simply view the output.

We find that we can achieve 97.5% accuracy on the test set, with a network that is about 1/8th the size of the fully connected network we used before which achieved 95.5% accuracy.