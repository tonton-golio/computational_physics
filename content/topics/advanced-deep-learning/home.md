# Advanced Deep Learning

## Roadmap

This course takes you on a journey from a single artificial neuron to the cutting edge of deep learning theory. Here is the path:

**A single neuron** learns to draw a line between two classes. Stack many of them into a **multilayer network** and you can approximate any function. Give the network **eyes** with convolutional filters and it starts to see edges, textures, and objects. Teach it to label **every pixel** with a U-shaped encoder-decoder, and it can outline a tumor in a brain scan. Now give it the tools to **design and train** properly: the right optimizer, the right initialization, the right regularization. With those tools in hand, let the network **dream**: a variational autoencoder learns a smooth landscape of possibilities, while a GAN pits a forger against a detective until the forgeries become indistinguishable from reality. Replace convolutions with **attention** and the network can read a sentence, translate a language, or caption an image. Finally, step back and ask the deepest question: **why does any of this work at all?** That is the theory.

$$
\text{Neuron} \;\to\; \text{MLP} \;\to\; \text{CNN} \;\to\; \text{U-Net} \;\to\; \text{Training toolkit} \;\to\; \text{VAE / GAN} \;\to\; \text{Transformer} \;\to\; \text{Theory}
$$

## Course overview

This course covers **state-of-the-art methods** in deep learning, including modern architectures, training techniques, theoretical foundations, and open research questions. Topics evolve yearly to reflect the rapidly advancing field.

- Architecture: from CNNs and autoencoders to transformers and diffusion models.
- Design: optimization algorithms, regularization strategies, and hyperparameter tuning.
- Analysis: generalization bounds, loss landscapes, and interpretability.
- Theory: approximation theory, neural tangent kernels, and adversarial robustness.

## Why this topic matters

- Deep learning has achieved superhuman performance in vision, language, and game-playing.
- Understanding *why* these models work (and when they fail) is an active research frontier.
- Designing training procedures and architectures requires principled methodology, not just trial and error.
- Adversarial robustness and uncertainty quantification are critical for deploying models safely.

## Key mathematical ideas

- Backpropagation and automatic differentiation.
- Optimization landscapes: convexity, saddle points, and implicit regularization.
- Generalization theory: PAC-Bayes bounds, double descent, and overparameterization.
- Information-theoretic perspectives on representation learning.
- Attention mechanisms and self-supervised learning objectives.

## Prerequisites

- Machine Learning A or equivalent introduction.
- Deep Learning fundamentals: feedforward networks, CNNs, backpropagation.
- Strong Python and PyTorch/TensorFlow skills.
- Linear algebra, multivariate calculus, and probability.

## Recommended reading

- Goodfellow, Bengio, and Courville, *Deep Learning*.
- Research papers from NeurIPS, ICML, and ICLR.
- Course notes announced each semester.

## Learning trajectory

This module is structured to build concepts progressively:

1. **Artificial neural networks**: Perceptrons, activation functions, MLPs, backpropagation, and the universal approximation theorem. You start with a single neuron and build up to networks that can learn any function.
2. **Convolutional neural networks**: Convolution operations, pooling, feature hierarchies, residual connections (ResNet), and transfer learning. The network gets eyes and learns to see.
3. **U-Net and segmentation**: Encoder-decoder with skip connections, Dice loss, and medical imaging applications. The natural next step after CNNs: pixel-level understanding using the same building blocks with a clever U-shaped pipe.
4. **Design and optimization**: SGD, Adam, learning rate schedules, regularization (dropout, batch norm), and initialization. The professional toolkit you need before building anything more ambitious.
5. **Variational autoencoders**: Probabilistic encoder-decoder, ELBO derivation, reparameterization trick, and latent space structure. The network learns to dream by mapping data onto a smooth landscape of possibilities.
6. **Generative adversarial networks**: Adversarial training, DCGAN, WGAN, mode collapse, and evaluation metrics. A forger and a detective train together until the forgeries are perfect.
7. **Transformers**: Attention mechanisms, multi-head attention, BERT, GPT, vision transformers, and scaling laws. The architecture that replaced everything by letting every part of the input talk to every other part.
8. **Analysis and theory**: Generalization bounds, double descent, neural tangent kernels, loss landscapes, adversarial robustness, and interpretability. The deepest question: why does deep learning work at all?

## Visual and simulation gallery

[[figure adl-mlops-loop]]

[[figure adl-cnn-feature-map]]
