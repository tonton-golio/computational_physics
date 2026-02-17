# Variational Autoencoders

## Introduction

**Variational autoencoders** (VAEs) are symmetric encoder-decoder neural networks used for data compression, dimensionality reduction, image generation, and intrinsic dimensionality estimation.

A standard autoencoder maps input data to a fixed-dimensional latent space through an encoder and reconstructs the input from the latent representation through a decoder. The objective is to minimize the reconstruction loss.

A VAE extends this by adding a **probabilistic layer** to the encoder that models the probability distribution of the latent space. This regularization ensures the latent space is smooth and continuous, enabling meaningful interpolation and generation.

[[simulation adl-pca-demo]]
[[simulation adl-latent-interpolation]]

## VAE Model (Fully Connected)

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 8)
        self.fc3 = nn.Linear(8, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc__ = nn.Linear(400, 400)

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h1 = self.relu(self.fc__(h1))
        return self.fc2(h1)

    def reparameterize(self, mu):
        std = torch.exp(0.5*mu)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h3 = self.relu(self.fc__(h3))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu)
        return self.decode(z), mu
```

## VAE with Convolutional Layers

```python
class VAE_with_conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv2_ = nn.Conv2d(32, 16, 3, padding=1)
        self.conv1_ = nn.Conv2d(16, 1, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 4)
        self.fc4_ = nn.Linear(2, 16)
        self.fc3_ = nn.Linear(16, 64)
        self.fc2_ = nn.Linear(64, 256)
        self.fc1_ = nn.Linear(256, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))
        h1 = self.relu(self.conv2_(h1))
        h1 = self.conv1_(h1)
        h1 = self.flatten(h1)
        h1 = self.relu(self.fc1(h1))
        h1 = self.relu(self.fc2(h1))
        h1 = self.relu(self.fc3(h1))
        return self.fc4(h1)

    def reparameterize(self, mu_std):
        std = mu_std[:, 2:]
        mu = mu_std[:, :2]
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu(self.fc4_(z))
        h3 = self.relu(self.fc3_(h3))
        h3 = self.relu(self.fc2_(h3))
        h3 = self.fc1_(h3)
        h3 = h3.view(-1, 1, 28, 28)
        return self.sigmoid(h3)

    def forward(self, x):
        mu_std = self.encode(x)
        z = self.reparameterize(mu_std)
        return self.decode(z), mu_std
```

## Latent Space Visualization

We take a single batch, encode all images, and examine where they fall in the latent space. By mapping the latent space to 2D using PCA, we can visualize how different digits cluster. Selecting a point on this 2D map and decoding it produces a generated image, demonstrating the smooth interpolation property of the VAE latent space.
