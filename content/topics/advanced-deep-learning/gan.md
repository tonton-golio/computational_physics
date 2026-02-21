# Generative Adversarial Networks

## The adversarial framework

Picture a boxing ring with two fighters. In one corner stands the **forger** — a network that takes random noise and tries to paint a convincing fake image. In the other corner stands the **detective** — a network that looks at images and tries to figure out which ones are real and which are forgeries. They train simultaneously, each getting better in response to the other. The forger studies why the detective caught him and paints more convincingly next time. The detective studies the forger's latest tricks and sharpens her eye. Round after round, the forgeries become more and more indistinguishable from reality.

That is a **generative adversarial network** (GAN). The **generator** $G$ maps random noise $\mathbf{z} \sim p_z(\mathbf{z})$ to synthetic data $G(\mathbf{z})$, while the **discriminator** $D$ tries to distinguish real data from generated samples. Training proceeds as a minimax game:

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))].
$$

At the Nash equilibrium, the generator produces samples indistinguishable from real data, and the discriminator outputs $D(\mathbf{x}) = 1/2$ everywhere — it genuinely cannot tell the difference. In practice, training alternates between updating $D$ (sharpen the detective) and updating $G$ (improve the forger).

## Architecture

A basic GAN for image generation uses:

* **Generator**: Takes a latent vector $\mathbf{z} \in \mathbb{R}^{d}$ (typically $d = 100$) and maps it through fully connected or transposed convolutional layers to produce an image. Batch normalization and ReLU activations are standard in intermediate layers, with a tanh output.

* **Discriminator**: Takes an image and outputs a single scalar (real/fake probability) through convolutional layers with LeakyReLU activations and a sigmoid output.

**DCGAN** (Deep Convolutional GAN) established key architectural guidelines: replace pooling with strided convolutions, use batch normalization in both networks, remove fully connected hidden layers, and use ReLU in the generator but LeakyReLU in the discriminator.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 784), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)
```
<!--code-toggle-->
```pseudocode
CLASS Generator:
    INIT(latent_dim=100):
        net = SEQUENTIAL(
            LINEAR(latent_dim, 256), BATCH_NORM(256), RELU,
            LINEAR(256, 512), BATCH_NORM(512), RELU,
            LINEAR(512, 784), TANH
        )
    FORWARD(z):
        RETURN RESHAPE(net(z), shape=(-1, 1, 28, 28))
```

## Training challenges

GAN training is notoriously unstable. The two-player game introduces failure modes that single-loss training never encounters:

**Mode collapse** happens when the forger discovers that one particular type of forgery reliably fools the detective, and stops trying anything else. The generator produces only a few distinct outputs, ignoring the diversity of the training distribution. It is like a forger who can perfectly copy one painting but cannot paint anything else.

**Vanishing gradients** happen when the detective becomes too good too fast. If $D(G(\mathbf{z})) \approx 0$ for all generated samples, the gradient of $\log(1 - D(G(\mathbf{z})))$ vanishes, and the generator gets no useful learning signal. The forger has been so thoroughly defeated that she cannot even tell *how* to improve. A practical fix is to train $G$ to maximize $\log D(G(\mathbf{z}))$ instead.

**Training instability** means the two-player game may not converge at all. The losses oscillate rather than decrease, with the generator and discriminator taking turns dominating.

[[simulation adl-gan-training-dynamics]]

## Improved loss functions

**Wasserstein GAN** (WGAN) addresses a limitation of the original saturating loss, which corresponds to minimizing the Jensen-Shannon divergence in the optimal-discriminator limit. (The non-saturating variant used in practice — training $G$ to maximize $\log D(G(\mathbf{z}))$ — has a different interpretation.) The JS divergence saturates to a constant $\log 2$ when the real and generated distributions have disjoint supports, which starves the generator of gradient signal. WGAN replaces it with the Earth Mover (Wasserstein-1) distance — a measure of how much "work" is needed to transform one distribution into another. The discriminator (called a "critic") outputs an unbounded score:

$$
\min_G \max_{D \in \mathcal{D}} \; \mathbb{E}_{\mathbf{x}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))],
$$

where $\mathcal{D}$ is the set of 1-Lipschitz functions. The Lipschitz constraint is enforced via **gradient penalty** (WGAN-GP):

$$
\lambda \, \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2],
$$

where $\hat{\mathbf{x}}$ is interpolated between real and generated samples. WGAN provides more meaningful loss curves (the loss actually correlates with sample quality) and significantly more stable training.

## Conditional GANs

**Conditional GANs** (cGAN) give the generator and discriminator additional information $\mathbf{y}$ (e.g., class labels):

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x}}[\log D(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z}, \mathbf{y}), \mathbf{y}))].
$$

This enables controlled generation: producing images of a specific digit, translating between image domains (pix2pix), or generating data with desired physical properties.

## VAE vs GAN: two philosophies of generation

VAEs and GANs represent fundamentally different approaches to the same problem. A VAE is a careful cartographer: it draws a smooth, continuous map of the data landscape and can reliably generate plausible outputs from any point on that map. The price is that its outputs tend to be blurry — the smoothness averages out sharp details.

A GAN is an arms race: the forger-detective competition drives the generator to produce crisp, realistic outputs, but the process is unstable and there is no guarantee of covering the full diversity of the data. The GAN may produce stunning individual samples while ignoring entire regions of the data distribution.

In terms of training, VAEs optimize a single stable loss function. GANs balance a two-player game that can oscillate or collapse. VAEs give you a tractable lower bound on the data likelihood; GANs give you no likelihood estimate at all. VAEs produce smooth latent spaces by design; GANs require careful tuning to achieve meaningful interpolation.

In practice, VAEs and GANs are complementary. Hybrid approaches like VAE-GAN combine the structured latent space of VAEs with the sharp outputs of GANs.

## What if the discriminator were perfect from the start?

If we gave the discriminator an oracle that perfectly distinguished real from fake, the generator would receive zero useful gradient — every generated sample would be instantly rejected with maximum confidence. The generator would learn nothing. The magic of GANs requires that both players are imperfect and improve together. The discriminator must be good enough to teach the generator but not so good that it silences it.

## Applications

GANs have found applications across science and engineering:

* **Image synthesis**: StyleGAN generates photorealistic faces by controlling style at different spatial scales.
* **Data augmentation**: Generating synthetic training data when real samples are scarce.
* **Super-resolution**: SRGAN recovers high-resolution images from low-resolution inputs.
* **Physics simulations**: Fast surrogate generators for expensive Monte Carlo simulations in particle physics and cosmology.
* **Anomaly detection**: The discriminator score identifies out-of-distribution samples.

## Evaluation metrics

Evaluating generative models is inherently difficult since we lack ground truth for the generated distribution. Common metrics include:

* **Frechet Inception Distance** (FID): Measures the distance between feature distributions of real and generated images using the Inception network. It computes the Frechet distance between two multivariate Gaussians fitted to the features: $\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$. Lower is better.
* **Inception Score** (IS): Evaluates both quality (sharp class predictions) and diversity (uniform class distribution). Higher is better.
* **Visual inspection**: Remains important for catching artifacts that quantitative metrics may miss. Numbers lie; your eyes usually do not.

## Big Ideas

* Mode collapse is what happens when the forger discovers a single forgery that reliably fools the detective and stops trying anything else — the game has a Nash equilibrium in principle, but practice offers many worse equilibria to fall into first.
* The original saturating GAN loss corresponds to minimizing the Jensen-Shannon divergence (in the optimal-discriminator limit), which saturates to a constant when the distributions do not overlap — Wasserstein GAN fixes this by using a distance that keeps providing signal even when the two distributions are far apart.
* The discriminator's job is not to win — it is to be helpful to the generator. A discriminator that destroys the generator too quickly leaves no gradient; a weak discriminator teaches nothing. The useful regime is always in between.
* FID captures both quality and diversity in one number, but it measures them in the feature space of a pre-trained classifier — which means it inherits all the biases of that classifier about what counts as realistic.

## What Comes Next

GANs produce sharp, realistic outputs — but their training is chaotic and they offer no way to measure how likely the data is under the model. The next lesson introduces **variational autoencoders**, which take a principled probabilistic approach to the same problem. Instead of an adversarial game, the VAE learns a smooth latent space through a single stable loss function derived from probability theory. The tradeoff: VAEs gain mathematical elegance and smooth interpolation at the cost of blurrier outputs — exactly the sharpness-smoothness tension that defines modern generative modeling.

## Check Your Understanding

1. Mode collapse occurs when the generator produces only a few distinct outputs. Why does the standard GAN loss not directly penalize this lack of diversity? What property of the minimax objective allows a generator to "win" by collapsing to a single mode?
2. In WGAN, the discriminator is replaced by a "critic" that outputs an unbounded real number rather than a probability. The critic must satisfy a Lipschitz constraint. Why does relaxing the output to be unbounded help, and what would happen without the Lipschitz constraint?
3. Evaluating GANs is hard because we cannot compute the likelihood of test data. FID measures the distance between real and generated feature distributions. What kind of generator failure would FID fail to detect?

## Challenge

Design a physics-motivated application for a GAN. Suppose you are simulating particle physics collisions: the full simulator takes minutes per event, but you need millions of events. Train a GAN to be a fast surrogate generator — the generator replaces the simulator, and the discriminator is trained on real simulation outputs. Think carefully about: how do you condition the GAN on input physics parameters (energy, collision type)? How do you validate that the generated events match the physics of the simulator, not just the visual appearance? What does mode collapse look like in this physical context, and how would you detect it?

