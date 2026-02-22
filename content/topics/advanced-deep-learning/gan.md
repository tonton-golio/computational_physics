# Generative Adversarial Networks

## The adversarial framework

Picture a boxing ring with two fighters. In one corner stands the **forger** -- a network that takes random noise and tries to paint a convincing fake image. In the other corner stands the **detective** -- a network that looks at images and tries to figure out which ones are real and which are forgeries. They train simultaneously, each getting better in response to the other. The forger studies why the detective caught him and paints more convincingly next time. The detective studies the forger's latest tricks and sharpens her eye. Round after round, the forgeries become indistinguishable from reality.

That's a **generative adversarial network** (GAN). The **generator** $G$ maps random noise $\mathbf{z} \sim p_z(\mathbf{z})$ to synthetic data $G(\mathbf{z})$, while the **discriminator** $D$ tries to distinguish real from generated samples. Training proceeds as a minimax game:

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))].
$$

At the Nash equilibrium, the generator produces samples indistinguishable from real data, and the discriminator outputs $D(\mathbf{x}) = 1/2$ everywhere -- it genuinely can't tell the difference.

## Architecture

A basic GAN for image generation uses:

* **Generator**: Takes a latent vector $\mathbf{z} \in \mathbb{R}^{d}$ (typically $d = 100$) and maps it through transposed convolutional layers to produce an image. Batch normalization and ReLU in intermediate layers, tanh output.
* **Discriminator**: Takes an image and outputs a single scalar (real/fake probability) through convolutional layers with LeakyReLU and a sigmoid output.

**DCGAN** established the key guidelines: replace pooling with strided convolutions, use batch normalization, remove fully connected hidden layers, ReLU in the generator, LeakyReLU in the discriminator.

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

GAN training is notoriously unstable. The two-player game introduces failure modes that single-loss training never encounters.

**Mode collapse** is the big one. It happens when the forger discovers that one particular type of forgery reliably fools the detective, and stops trying anything else. It's like a forger who can perfectly copy one Rembrandt but can't paint anything else. The generator produces only a few distinct outputs, ignoring the rich diversity of the training distribution.

**Vanishing gradients** happen when the detective becomes too good too fast. If $D(G(\mathbf{z})) \approx 0$ for all generated samples, the generator gets no useful learning signal. The forger has been so thoroughly defeated she can't even tell *how* to improve. A practical fix: train $G$ to maximize $\log D(G(\mathbf{z}))$ instead.

**Training instability** means the two-player game may not converge at all. The losses oscillate, with generator and discriminator taking turns dominating.

[[simulation adl-gan-training-dynamics]]

## Improved loss functions

**Wasserstein GAN** (WGAN) fixes a deep problem with the original loss. The Jensen-Shannon divergence saturates to a constant when the real and generated distributions don't overlap -- which starves the generator of gradient signal. WGAN replaces it with the Earth Mover distance, a measure of how much "work" is needed to transform one distribution into another:

$$
\min_G \max_{D \in \mathcal{D}} \; \mathbb{E}_{\mathbf{x}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))],
$$

where $\mathcal{D}$ is the set of 1-Lipschitz functions. The Lipschitz constraint is enforced via **gradient penalty** (WGAN-GP). The result: more meaningful loss curves (the loss actually correlates with sample quality) and significantly more stable training.

## Conditional GANs

**Conditional GANs** give both networks additional information $\mathbf{y}$ (like class labels), enabling controlled generation -- producing images of a specific digit, translating between image domains, or generating data with desired physical properties.

## VAE vs GAN: the punchline

VAE draws a smooth map. GAN runs an arms race. The VAE's outputs tend to be blurry because smoothness averages out sharp details. The GAN's outputs tend to be crisp but may ignore entire regions of the data. VAEs optimize a single stable loss; GANs balance a two-player game that can oscillate or collapse. In practice they're complementary -- hybrid approaches like VAE-GAN combine the structured latent space of one with the sharp outputs of the other.

## Applications

GANs have found applications across science and engineering:

* **Image synthesis**: StyleGAN generates photorealistic faces by controlling style at different spatial scales.
* **Data augmentation**: Generating synthetic training data when real samples are scarce.
* **Super-resolution**: SRGAN recovers high-resolution images from low-resolution inputs.
* **Physics simulations**: Fast surrogate generators for expensive Monte Carlo simulations in particle physics and cosmology.

## Big Ideas

* Mode collapse is the forger discovering a single forgery that reliably fools the detective and stopping there -- the game has a Nash equilibrium in principle, but practice offers many worse equilibria to fall into first.
* The discriminator's job is not to win -- it's to be *helpful*. A discriminator that crushes the generator too quickly leaves no gradient; a weak one teaches nothing. The useful regime is always in between.
* WGAN keeps providing gradient signal even when the two distributions are far apart, which is why it stabilized training where the original loss couldn't.

## What Comes Next

GANs produce sharp, realistic outputs -- but their training is chaotic and they offer no way to measure how likely the data is under the model. The next lesson introduces **variational autoencoders**, which take a principled probabilistic approach. Instead of an adversarial game, the VAE learns a smooth latent space through a single stable loss derived from probability theory. The tradeoff: mathematical elegance and smooth interpolation at the cost of blurrier outputs.
