# Variational Autoencoders

## Standard autoencoders

An **autoencoder** compresses input $\mathbf{x}$ to a low-dimensional latent representation $\mathbf{z}$ through an encoder $f_\phi$, then reconstructs the input through a decoder $g_\theta$:

$$
\mathbf{z} = f_\phi(\mathbf{x}), \qquad \hat{\mathbf{x}} = g_\theta(\mathbf{z}).
$$

Training minimizes reconstruction loss: $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$.

The problem: the latent space of a standard autoencoder is a mess. It is **discontinuous and unstructured** — nearby points in latent space may decode to wildly different outputs, and large regions of the latent space correspond to nothing meaningful at all. If you pick a random point in latent space and try to decode it, you will likely get garbage. The autoencoder learned to compress and decompress, but it did not learn a *map* of all possible outputs.

## The VAE idea

A **variational autoencoder** (VAE) fixes this by changing one thing: instead of mapping each input to a single point in latent space, the encoder outputs the parameters of a probability distribution — a little cloud of uncertainty around where the input *should* be in latent space:

$$
q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}))).
$$

The decoder then reconstructs from a sample $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$. By forcing these little clouds to stay close to a standard normal prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$, the VAE ensures that the latent space is smooth and well-organized. Every region of latent space decodes to something meaningful, and walking smoothly through latent space produces smooth transformations in the output.

[[simulation adl-pca-demo]]

## ELBO derivation

Why does the VAE loss function look the way it does? We want to maximize the marginal log-likelihood $\log p_\theta(\mathbf{x})$ — the probability that our model assigns to the data we actually observe. But computing this directly requires integrating over all possible latent codes $\mathbf{z}$, which is intractable.

The trick is to derive a tractable lower bound. Starting from the log-marginal likelihood and introducing an approximate posterior $q_\phi(\mathbf{z}|\mathbf{x})$:

$$
\log p_\theta(\mathbf{x}) = \log \int p_\theta(\mathbf{x}|\mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}.
$$

Applying Jensen's inequality (or equivalently, decomposing the KL divergence):

$$
\log p_\theta(\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{ELBO}} + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p_\theta(\mathbf{z}|\mathbf{x})).
$$

Since the last KL term is non-negative, the **Evidence Lower Bound** (ELBO) is a lower bound on the log-likelihood. Maximizing the ELBO simultaneously:
1. Maximizes the expected reconstruction quality (first term).
2. Minimizes the gap between the approximate and true posterior (last KL term vanishes when $q_\phi = p_\theta(\mathbf{z}|\mathbf{x})$).

The loss function is therefore:

$$
\mathcal{L}_{\text{VAE}} = -\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] + D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})).
$$

The first term says "reconstruct well." The second term says "do not stray too far from the prior." Together they produce a model that generates well and has a smooth latent space.

## The reparameterization trick

The reparameterization trick is the cleverest accounting hack in deep learning. Here is the problem: the encoder outputs a distribution, and the decoder needs a sample from that distribution. But backprop can flow through addition and multiplication — not through "sample from $\mathcal{N}(\mu, \sigma)$." The gradient of "pick a random number" with respect to the distribution parameters is undefined.

The trick: we move the randomness **outside** the computation graph that depends on the network parameters. Instead of sampling directly from the learned distribution, we sample a "frozen" piece of noise $\boldsymbol{\epsilon}$ from a fixed standard normal, and then *deterministically* transform it using the learned parameters:

$$
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$

The sampler is now a fixed random-number generator, not a stochastic node that blocks gradients. All the randomness lives in $\boldsymbol{\epsilon}$, which does not depend on any learnable parameters. The gradient flows cleanly through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$, and we can train the whole system end-to-end with standard backpropagation. We rewrote the sampling step so backprop never sees the random node.

## KL divergence term

For Gaussian encoder and standard normal prior, the KL divergence has a closed-form solution:

$$
D_{\text{KL}}(q_\phi \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right).
$$

What is this term actually doing? It is the universe telling the network "do not be too sure of yourself." Without the KL term, the encoder would collapse each input to a single point (zero variance) at some arbitrary location — perfectly memorizing each training example but creating a latent space full of gaps. The KL penalty forces each encoding to be a spread-out cloud that overlaps with the prior, filling the latent space with meaning.

Think of the KL term as a parking-lot attendant who forces every car (encoding) to leave a little space around it. Without the attendant, every car parks exactly on top of a training example and the lot becomes useless for new arrivals.

## What if we didn't have the KL term?

Without the KL regularizer, the VAE degenerates into a regular autoencoder. The encoder would learn to place each training example at a unique, precise point in latent space with zero variance. Reconstruction would be perfect, but sampling would be useless — random points in latent space would decode to nothing coherent. The KL term is the price of generativity.

## Reconstruction vs KL tradeoff: Beta-VAE

The standard VAE loss weights reconstruction and KL terms equally. **Beta-VAE** introduces an explicit tradeoff:

$$
\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[-\log p_\theta(\mathbf{x}|\mathbf{z})] + \beta \cdot D_{\text{KL}}(q_\phi \| p).
$$

* $\beta > 1$: Stronger regularization, encouraging more disentangled latent representations at the cost of reconstruction quality. The network is forced to organize its latent space more carefully.
* $\beta < 1$: Better reconstruction but a less structured latent space.

## VAE model (fully connected)

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

## Latent space properties

A well-trained VAE exhibits several desirable latent space properties:

* **Smoothness**: Nearby points in latent space decode to similar outputs. Walk from one digit to another and you see a continuous morphing, not a sudden jump.
* **Completeness**: Every point sampled from the prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ decodes to a plausible output. There are no dead zones.
* **Disentanglement**: Individual latent dimensions may correspond to interpretable factors of variation (e.g., digit identity, slant, thickness). Twist one knob and only one thing changes.

[[simulation adl-latent-interpolation]]

## Latent space visualization

We take a single batch, encode all images, and examine where they fall in the latent space. By mapping the latent space to 2D using PCA, we can visualize how different digits cluster. Selecting a point on this 2D map and decoding it produces a generated image, demonstrating the smooth interpolation property of the VAE latent space.

## Big Ideas

* The reparameterization trick is a sleight of hand that moves randomness out of the parameters and into a fixed noise variable — backpropagation can flow through deterministic operations but not through sampling, so you change where the sampling happens.
* The KL term is a leash on the encoder: without it, each training example retreats to its own isolated point in latent space, and the space between them becomes a wasteland where the decoder has never been trained to operate.
* A smooth latent space is not an aesthetic preference — it is what makes interpolation and generation meaningful. Walking from one encoding to another should produce a path through plausible data, not a teleportation through garbage.
* Beta-VAE reveals a tension baked into the architecture: tighter KL regularization forces more disentangled representations but at the cost of blurrier reconstructions — you cannot have perfect fidelity and perfect organization at the same time.

## What Comes Next

The VAE gives you a smooth latent space and principled generation, but the blurriness of its outputs reveals a fundamental limitation of pixel-level reconstruction losses. The next lesson introduces the **U-Net architecture**, which takes the encoder-decoder structure you just learned and applies it to a different problem: **image segmentation**. Instead of generating images from latent codes, the U-Net maps images to pixel-level predictions — assigning a class label to every single pixel. The skip connections that made residual networks work reappear here in a new guise, carrying fine spatial detail across the compression bottleneck.

## Check Your Understanding

1. A standard autoencoder encodes inputs to single points in latent space and achieves zero reconstruction error on training data. Why can you not simply sample a random point from this latent space to generate a new image?
2. The ELBO is a lower bound on the log-likelihood of the data. Maximizing the ELBO does not guarantee maximizing the true likelihood — there is always a gap equal to the KL divergence between the approximate and true posterior. What does it mean in practice if this gap is large?
3. Walking in a straight line through the latent space of a well-trained VAE produces smooth morphs between digits. Would you expect the same to be true for a standard autoencoder with the same architecture? Design a simple experiment to test this.

## Challenge

The standard VAE uses a Gaussian prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and a Gaussian approximate posterior. This forces all classes into a single, undifferentiated ball in latent space. Design and implement a **conditional VAE** where the prior depends on the class label — e.g., each digit class has its own Gaussian prior centered at a different location. Does this produce more separable latent representations? Does it improve generation quality for specific classes? How does the reconstruction-vs-KL tradeoff change when the prior is no longer a single fixed Gaussian?

