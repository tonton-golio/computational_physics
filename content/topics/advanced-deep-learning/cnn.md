# Convolutional Neural Networks

## Introduction

**Convolutional neural networks** (CNNs) are better suited for image recognition than fully connected networks because they exploit spatial structure and recognize patterns at multiple scales.

[[simulation adl-convolution-demo]]

## Kernel Types

We may choose from a variety of convolution kernels:
- **Averaging**: `np.ones(k,k)/k**2` (simple blur)
- **Gaussian blur**: Weighted average with Gaussian falloff
- **Dilated (Atrous) convolution**: Expands the receptive field without increasing parameters

## CNN Model

```python
class ConvolutionalNetwork(nn.Module):
    '''
    Architecture:
    - convolutional layer -> max pooling
    - convolutional layer -> max pooling
    - fully connected layer -> output layer

    Takes a 28x28 image and outputs a 10-dim vector of logits.
    '''
    def __init__(self, im_shape=(1, 28, 28), n_classes=10, batch_size=100):
        super().__init__()
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.n_classes = n_classes

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

## Evaluation

The CNN achieves approximately 97.5% accuracy on the MNIST test set, with a network about 1/8th the size of the fully connected MLP (which achieved 95.5%). This demonstrates the efficiency of exploiting spatial structure through convolutions.
