# Variational Autoencoders

## Standard autoencoders

An **autoencoder** compresses input $\mathbf{x}$ to a low-dimensional latent representation $\mathbf{z}$ through an encoder, then reconstructs through a decoder:

$$
\mathbf{z} = f_\phi(\mathbf{x}), \qquad \hat{\mathbf{x}} = g_\theta(\mathbf{z}).
$$

Training minimizes reconstruction loss: $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$.

The problem? The latent space of a standard autoencoder is a mess. It's **discontinuous and unstructured** -- nearby points may decode to wildly different outputs, and large regions correspond to nothing meaningful at all. Pick a random point and try to decode it: you'll get garbage. The autoencoder learned to compress and decompress, but it didn't learn a *map* of all possible outputs.

## The VAE idea

A **variational autoencoder** (VAE) fixes this by changing one thing: instead of mapping each input to a single point in latent space, the encoder outputs the parameters of a probability distribution -- a little cloud of uncertainty around where the input *should* be:

$$
q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))).
$$

By forcing these clouds to stay close to a standard normal prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, the VAE ensures the latent space is smooth and well-organized. Every region decodes to something meaningful. Walking smoothly through latent space produces smooth transformations in the output.

## The ELBO

We want to maximize $\log p_\theta(\mathbf{x})$ -- the probability our model assigns to the data we observe. But computing this directly requires integrating over all possible latent codes, which is intractable.

The trick: derive a tractable lower bound. The **Evidence Lower Bound** (ELBO) decomposes into two terms:

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})).
$$

The first term says "reconstruct well." The second says "don't stray too far from the prior." Together they produce a model that generates well and has a smooth latent space.

## Reconstruction vs KL: Beta-VAE

And here's a knob you can turn. **Beta-VAE** introduces an explicit tradeoff:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[-\log p_\theta(\mathbf{x}|\mathbf{z})] + \beta \cdot D_{\text{KL}}(q_\phi \| p).
$$

$\beta > 1$ forces more disentangled latent representations at the cost of blurrier reconstructions. $\beta < 1$ gives better reconstruction but a messier latent space. You can't have perfect fidelity and perfect organization at the same time -- that tension is baked into the architecture.

## The reparameterization trick

This is the single most clever trick in the course. Give it room, because it's beautiful.

Here's the problem: the encoder outputs a distribution, and the decoder needs a sample from it. But backprop can flow through addition and multiplication -- not through "sample from $\mathcal{N}(\mu, \sigma)$." The gradient of "pick a random number" with respect to the distribution parameters is undefined.

The trick: we sneak the randomness *outside* so the gradient can still flow through. Instead of sampling directly from the learned distribution, we sample a frozen piece of noise $\boldsymbol{\epsilon}$ from a fixed standard normal, and then *deterministically* transform it:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

All the randomness lives in $\boldsymbol{\epsilon}$, which doesn't depend on any learnable parameters. The gradient flows cleanly through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, and we train the whole system end-to-end with standard backprop. We rewrote the sampling step so backprop never even sees the random node. That's it. That's the whole trick.

## KL divergence term

For Gaussian encoder and standard normal prior, the KL divergence has a closed form:

$$
D_{\text{KL}}(q_\phi \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right).
$$

What's this term actually doing? It's the universe telling the network "don't be too sure of yourself." Without it, the encoder would collapse each input to a single point at some arbitrary location -- perfectly memorizing each training example but creating a latent space full of gaps. The KL penalty forces each encoding to be a spread-out cloud that overlaps with the prior, filling the latent space with meaning.

## VAE model

```python
class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 8)
        self.fc_logvar = nn.Linear(400, 8)
        self.fc3 = nn.Linear(8, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```
<!--code-toggle-->
```pseudocode
CLASS VAE:
    INIT():
        fc1     = LINEAR(784, 400)
        fc_mu   = LINEAR(400, 8)
        fc_logvar = LINEAR(400, 8)
        fc3     = LINEAR(8, 400)
        fc4     = LINEAR(400, 784)

    ENCODE(x):
        h = RELU(fc1(x))
        RETURN fc_mu(h), fc_logvar(h)

    REPARAMETERIZE(mu, logvar):
        std = EXP(0.5 * logvar)
        eps = SAMPLE_NORMAL(shape=std.shape)
        RETURN mu + eps * std

    DECODE(z):
        h = RELU(fc3(z))
        RETURN SIGMOID(fc4(h))

    FORWARD(x):
        mu, logvar = ENCODE(FLATTEN(x))
        z = REPARAMETERIZE(mu, logvar)
        RETURN DECODE(z), mu, logvar
```

The VAE loss combines reconstruction and KL terms:

```python
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```
<!--code-toggle-->
```pseudocode
FUNCTION vae_loss(recon_x, x, mu, logvar):
    recon_loss = BINARY_CROSS_ENTROPY(recon_x, FLATTEN(x), reduction="sum")
    kl_loss = -0.5 * SUM(1 + logvar - mu^2 - EXP(logvar))
    RETURN recon_loss + kl_loss
```

## The latent space

Walk through the latent space and the digits melt smoothly into one another -- a 3 becomes a 5 becomes an 8 in a continuous morphing that never passes through garbage. That smoothness is the whole point. Every point sampled from the prior decodes to something plausible. There are no dead zones. And individual latent dimensions may correspond to interpretable factors -- digit identity, slant, thickness. Twist one knob and only one thing changes.

## Big Ideas

* The reparameterization trick moves randomness out of the parameters and into a fixed noise variable -- backprop can flow through deterministic operations but not through sampling, so you change where the sampling happens.
* The KL term is a leash on the encoder: without it, each training example retreats to its own isolated point, and the space between them becomes a wasteland.
* A smooth latent space isn't an aesthetic preference -- it's what makes interpolation and generation meaningful.

## What Comes Next

The VAE gives you a smooth latent space and principled generation, but the blurriness of its outputs reveals a fundamental limitation of pixel-level reconstruction losses. The next lesson introduces the **U-Net architecture**, which takes the encoder-decoder structure you just learned and applies it to a different problem: **image segmentation**. Instead of generating images from latent codes, the U-Net maps images to pixel-level predictions -- assigning a class label to every single pixel.
