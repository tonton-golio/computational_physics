
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

KEY: VAE conv model
### The model
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
        #st.write(z.shape)
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




KEY: latent space
### The latent space
We take a single batch, encode all the images. Look where they fall in the latent space -> we then map this latent space to 2d with PCA. Selecting a point on this 2d map, we can then decode it to an image.
