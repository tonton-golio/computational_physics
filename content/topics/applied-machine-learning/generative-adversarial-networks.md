# Generative Adversarial Networks (Extension)

Generative adversarial networks (GANs) train two models in competition:
- a **generator** that creates synthetic samples,
- a **discriminator** that distinguishes real from generated samples.

## Adversarial objective (conceptual)
$$
\min_G \max_D \; \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
$$

This adversarial game encourages generated samples to match the target data distribution.

## Practical stability notes
- GAN training is sensitive to architecture and optimizer settings.
- Common issues: mode collapse, unstable gradients, discriminator overpowering.
- Common improvements: Wasserstein objectives, gradient penalties, spectral normalization.

## Cross-topic ownership
GANs are covered in depth in Advanced Deep Learning. Use this page as context and transition:
- `/topics/advanced-deep-learning/gan`

## Recommended use in this module
Treat GANs as a bridge from classical applied ML to modern deep generative modeling.
