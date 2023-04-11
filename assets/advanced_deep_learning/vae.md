
KEY: intro
Variation auto encoders are symmetric encoder decoder neural networks. They can be used for data compression, dimensionality reduction, and image generation, intrinsic dimensionality estimation, and probably other things.

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="https://miro.medium.com/v2/resize:fit:1100/format:webp/1*UdOybs9wOe3zW8vDAfj9VA@2x.png" width=500>
</div>

A normal autoencoder learns to map the input data to a fixed-dimensional latent space through an encoder network and then reconstructs the input data from the latent space through a decoder network. The objective of a normal autoencoder is to minimize the reconstruction loss between the input data and the reconstructed data.

In contrast, a VAE extends this by adding a probabilistic layer to the encoder that models the probability distribution of the latent space.


KEY: VAE model
### The model
```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 5)
        self.fc3 = nn.Linear(5, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc2(h1)
    
    def reparameterize(self, mu):
        std = torch.exp(0.5*mu)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu)
        return self.decode(z), mu
```


KEY: latent space
### The latent space
We take a single batch, encode all the images. Look where they fall in the latent space -> we then map this latent space to 2d with PCA. Selecting a point on this 2d map, we can then decode it to an image.
