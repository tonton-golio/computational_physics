# Generative Adversarial Networks (Extension)

Imagine two kids locked in an arms race. One is a forger who paints fake banknotes. The other is a detective who inspects them. Every time the detective catches a fake, the forger studies what gave it away and makes a better fake next time. Every time a fake slips past, the detective sharpens their eye. Over time, both get astonishingly good — and the fakes become nearly indistinguishable from the real thing. That is the core idea behind generative adversarial networks.

## The adversarial game

A GAN trains two neural networks simultaneously:

- A **generator** $G$ takes random noise $z$ and produces a synthetic sample $G(z)$. Its goal is to fool the discriminator.
- A **discriminator** $D$ receives both real samples from the training data and fakes from the generator. Its goal is to tell them apart.

The two play a min-max game:

$$
\min_G \max_D \; \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
$$

Think of this as a zero-sum tennis match. The discriminator tries to maximize its score (correctly classifying real vs. fake), while the generator tries to minimize it (making fakes that the discriminator calls real). At equilibrium — if you ever reach it — the generator produces samples so realistic that the discriminator can do no better than random guessing.

In plain English: the discriminator is rewarded twice — once for correctly saying "real" to real images, and once for correctly saying "fake" to generated images. The generator is only rewarded when the discriminator says "real" to a fake. Its entire job is to make the discriminator's second reward as small as possible — by becoming indistinguishable from reality.

## Mode collapse: when the forger gets lazy

Here is the most infamous failure mode. The forger discovers that painting only happy faces always fools the detective — so they never learn to paint landscapes, animals, or buildings. The generator has found a narrow pocket of "safe" outputs and stubbornly refuses to explore the full diversity of the data distribution.

This is **mode collapse**: the generator produces high-quality but low-diversity samples, capturing only a few modes of the real distribution. You asked for a model that can generate any face, and you got one that generates the same three faces over and over.

## Stabilizing training

GAN training is notoriously finicky. The generator and discriminator are locked in a delicate dance, and small imbalances can cascade.

**Wasserstein objective** replaces the log-probability game with a smoother distance metric (the earth-mover distance), giving the generator more useful gradients even when the discriminator is very confident.

**Gradient penalty** adds a regularization term that prevents the discriminator from becoming too sharp, keeping the gradients informative for the generator.

**Spectral normalization** constrains the discriminator's weights to control its Lipschitz constant, stabilizing the adversarial dynamics.

These are engineering solutions to a fundamental tension: the two networks must improve at roughly the same rate, or one overpowers the other and learning stalls.

## Where this goes deeper

This page gives you the intuition for adversarial training. GANs are a bridge from classical applied ML to the wild world of deep generative modeling. For architecture details (DCGAN, StyleGAN, conditional GANs), training recipes, and the relationship to other generative approaches (VAEs, diffusion models), see [Advanced Deep Learning — Generative Models](/topics/advanced-deep-learning/gan).

[[figure aml-gan-progression]]

[[simulation aml-gan-forger-arena]]

## Check your understanding

- Can you explain the GAN training loop to a friend using the forger-and-detective story?
- If you had to explain mode collapse in one drawing, what would it look like?
- What experiment would tell you whether your GAN is suffering from mode collapse rather than simply needing more training?
