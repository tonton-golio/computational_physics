# Generative Adversarial Networks

## The Adversarial Framework

A **generative adversarial network** (GAN) consists of two neural networks trained in opposition. The **generator** $G$ maps random noise $\mathbf{z} \sim p_z(\mathbf{z})$ to synthetic data $G(\mathbf{z})$, while the **discriminator** $D$ tries to distinguish real data from generated samples. Training proceeds as a minimax game:

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))].
$$

At the Nash equilibrium, the generator produces samples indistinguishable from real data, and the discriminator outputs $D(\mathbf{x}) = 1/2$ everywhere. In practice, training alternates between updating $D$ (to better classify real vs. fake) and updating $G$ (to better fool $D$).

## Architecture

A basic GAN for image generation uses:

- **Generator**: Takes a latent vector $\mathbf{z} \in \mathbb{R}^{d}$ (typically $d = 100$) and maps it through fully connected or transposed convolutional layers to produce an image. Batch normalization and ReLU activations are standard in intermediate layers, with a tanh output.

- **Discriminator**: Takes an image and outputs a single scalar (real/fake probability) through convolutional layers with LeakyReLU activations and a sigmoid output.

**DCGAN** (Deep Convolutional GAN) established key architectural guidelines: replace pooling with strided convolutions, use batch normalization in both networks, remove fully connected hidden layers, and use ReLU in the generator but LeakyReLU in the discriminator.

## Training Challenges

GAN training is notoriously unstable due to several failure modes:

- **Mode collapse**: The generator produces only a few distinct outputs, ignoring the diversity of the training distribution. The discriminator may oscillate between rejecting and accepting these modes.

- **Vanishing gradients**: If the discriminator becomes too strong, $D(G(\mathbf{z})) \approx 0$ and the gradient of $\log(1 - D(G(\mathbf{z})))$ vanishes, stalling generator learning. A practical fix is to train $G$ to maximize $\log D(G(\mathbf{z}))$ instead.

- **Training instability**: The two-player game may not converge, with loss oscillating rather than decreasing. Techniques like label smoothing, gradient penalty, and careful learning rate tuning help stabilize training.

## Improved Loss Functions

**Wasserstein GAN** (WGAN) replaces the JS divergence implicit in the original loss with the Earth Mover distance. The discriminator (called a "critic") outputs an unbounded score, and the loss becomes

$$
\min_G \max_{D \in \mathcal{D}} \; \mathbb{E}_{\mathbf{x}}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{z}}[D(G(\mathbf{z}))],
$$

where $\mathcal{D}$ is the set of 1-Lipschitz functions. The Lipschitz constraint is enforced via **gradient penalty** (WGAN-GP):

$$
\lambda \, \mathbb{E}_{\hat{\mathbf{x}}}[(\|\nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}})\|_2 - 1)^2],
$$

where $\hat{\mathbf{x}}$ is interpolated between real and generated samples. WGAN provides more meaningful loss curves and stable training.

## Conditional GANs

**Conditional GANs** (cGAN) condition both generator and discriminator on additional information $\mathbf{y}$ (e.g., class labels):

$$
\min_G \max_D \; \mathbb{E}_{\mathbf{x}}[\log D(\mathbf{x}, \mathbf{y})] + \mathbb{E}_{\mathbf{z}}[\log(1 - D(G(\mathbf{z}, \mathbf{y}), \mathbf{y}))].
$$

This enables controlled generation: producing images of a specific digit, translating between image domains (pix2pix), or generating data with desired physical properties.

## Applications

GANs have found applications across science and engineering:

- **Image synthesis**: StyleGAN generates photorealistic faces by controlling style at different spatial scales.
- **Data augmentation**: Generating synthetic training data when real samples are scarce.
- **Super-resolution**: SRGAN recovers high-resolution images from low-resolution inputs.
- **Physics simulations**: Fast surrogate generators for expensive Monte Carlo simulations in particle physics and cosmology.
- **Anomaly detection**: The discriminator score identifies out-of-distribution samples.

## Evaluation Metrics

Evaluating generative models is inherently difficult since we lack ground truth for the generated distribution. Common metrics include:

- **Frechet Inception Distance** (FID): Measures the distance between feature distributions of real and generated images. Lower is better.
- **Inception Score** (IS): Evaluates both quality (sharp class predictions) and diversity (uniform class distribution). Higher is better.
- **Visual inspection**: Remains important for catching artifacts that quantitative metrics may miss.

[[simulation adl-convolution-demo]]
