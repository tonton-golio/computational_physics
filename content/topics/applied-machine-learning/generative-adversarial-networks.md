# Generative Adversarial Networks (Extension)

Imagine two kids locked in an arms race. One is a forger who paints fake banknotes. The other is a detective who inspects them. Every time the detective catches a fake, the forger studies what gave it away and makes a better one. Every time a fake slips past, the detective sharpens their eye. Over time, both get astonishingly good -- and the fakes become nearly indistinguishable from the real thing. That is the core idea behind generative adversarial networks.

## The adversarial game

A GAN trains two neural networks simultaneously:

* A **generator** $G$ takes random noise $z$ and produces a synthetic sample $G(z)$. Its goal is to fool the discriminator.
* A **discriminator** $D$ receives both real samples and fakes. Its goal is to tell them apart.

The two play a min-max game:

$$
\min_G \max_D \; \mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]
$$

Think of it as a zero-sum tennis match. The discriminator tries to maximize its score (correctly classifying real vs. fake), while the generator tries to minimize it (making fakes the discriminator calls real). At equilibrium -- if you ever reach it -- the generator produces samples so realistic that the discriminator can do no better than random guessing.

## Mode collapse: when the forger gets lazy

Here is the most infamous failure mode. The forger discovers that painting only happy faces always fools the detective -- so they never learn to paint landscapes, animals, or buildings. The generator has found one narrow pocket of "safe" outputs and refuses to explore the full diversity of the data.

You asked for a model that can generate any face, and you got one that generates the same three faces over and over. Mode collapse is the GAN version of "the model found one answer that works and stopped exploring."

## So what did we really learn?

A GAN is a machine for turning a simple, known distribution (random noise) into a complex, unknown one (real data), with no explicit density to fit -- only a critic that says "real" or "fake."

Mode collapse reveals the core tension: the generator is rewarded for fooling the discriminator, not for being diverse. A clever forger who specializes in one convincing fake beats an honest forger who tries to cover everything.

GANs are historically important not because they are perfect, but because they taught us you can train a critic to tell you when something looks right. That idea is still spreading everywhere -- reinforcement learning from human feedback, learned metrics, adversarial robustness. Once you have internalized the forger-and-detective story, you will recognize it operating quietly in places that do not call themselves GANs at all.

## Challenge

Mode collapse is easier to demonstrate than to fix. Build a toy GAN that tries to learn a mixture of five well-separated Gaussians in 2D (small enough to train on a laptop in minutes). Train it with the standard binary cross-entropy objective and observe whether it collapses to fewer than five modes. Measure diversity quantitatively: fit a Gaussian mixture to the generator's samples and count how many modes it recovers. What hyperparameters (learning rate ratio between G and D, number of discriminator steps per generator step) most influence whether collapse occurs?
