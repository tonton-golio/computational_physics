# Convolutional Neural Networks

## Seeing edges by hand

Before any formulas, let us do a convolution by hand. Take a tiny 5x5 image — say, a white square on a black background. Now take a small 3x3 grid of numbers called a **Sobel filter** (it looks like $[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]$). Place this little grid on the top-left corner of the image and multiply each overlapping pair of numbers, then add them all up. That gives you one number. Slide the grid one pixel to the right and repeat. Keep sliding until you have covered the whole image. What you get is a new image where bright spots mark vertical edges — places where the intensity changes sharply from left to right. That sliding-and-multiplying operation is a **convolution**, and the little grid is a **kernel**.

The key insight: a neural network can *learn* what kernels to use. Instead of hand-designing a Sobel filter, the network discovers — through backpropagation — exactly which patterns to look for.

## Why convolutions?

Fully connected networks treat each pixel independently, ignoring the spatial structure of images. **Convolutional neural networks** (CNNs) exploit three key properties:

- **Translation equivariance**: A pattern detected in one part of the image can be recognized elsewhere without learning separate weights for each position. An edge is an edge, whether it appears in the top-left corner or the bottom-right.
- **Parameter sharing**: The same kernel weights are applied at every spatial location, dramatically reducing the parameter count.
- **Locality**: Each output depends only on a small region of the input, capturing local patterns before combining them into global features.

These inductive biases make CNNs far more efficient than MLPs for image tasks. A fully connected layer connecting two 28x28 feature maps would require $784^2 \approx 600{,}000$ parameters; a 3x3 convolutional layer needs only 9.

## The convolution operation

Imagine you are sliding a little magnifying glass over the picture and asking, at every spot, "How much does this tiny 3x3 window look like the pattern I am searching for?" That sliding multiplication is convolution. Formally, a 2D convolution slides a kernel $K$ of size $k \times k$ across an input feature map $I$, computing at each position:

$$
(I * K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i + m, j + n] \cdot K[m, n].
$$

Key parameters that control the sliding:
- **Stride**: The step size when sliding the kernel. Stride 2 halves the spatial dimensions.
- **Padding**: Adding zeros around the border. "Same" padding preserves spatial dimensions; "valid" padding shrinks them.
- **Dilation**: Inserting gaps between kernel elements to increase the receptive field without adding parameters.

[[simulation adl-convolution-demo]]

## Kernel types

Different kernels detect different things:
- **Averaging**: A uniform kernel that blurs the image by averaging nearby pixels.
- **Gaussian blur**: A weighted average with Gaussian falloff, producing smoother blurring.
- **Edge detection**: Kernels like Sobel or Laplacian that highlight intensity changes — exactly what we did by hand above.
- **Dilated (atrous) convolution**: Expands the receptive field without increasing parameters or pooling.

In a CNN, the network learns its own kernels through training. Early kernels tend to look like edge and texture detectors; deeper kernels respond to increasingly complex patterns.

## Feature hierarchies

Deep CNNs learn a hierarchy of features, building complexity layer by layer the way a painter builds a picture:
- **Early layers** detect low-level features: edges, corners, color gradients. These are the brushstrokes.
- **Middle layers** combine these into mid-level patterns: textures, object parts, shapes. An ear, a wheel, a petal.
- **Deep layers** recognize high-level concepts: objects, scenes, semantic categories. A face, a car, a flower.

This progression from simple to complex is analogous to the ventral visual stream in the brain, where visual processing progresses from V1 (edges) to IT cortex (object recognition). The network rediscovers a strategy that evolution took millions of years to build.

## What if we didn't have convolutions?

Without convolutions, you would need a fully connected layer for every pixel-to-pixel connection. A single layer processing a modest 224x224 RGB image would need $224 \times 224 \times 3 \times 224 \times 224 \times 3 \approx 7$ billion parameters — for *one* layer. And the network would have to learn separately that an edge in the top-left is the same concept as an edge in the bottom-right. Convolutions give you parameter sharing and spatial awareness for free.

## Pooling layers

**Pooling** reduces spatial dimensions and provides a degree of translation invariance:

- **Max pooling**: Takes the maximum value in each window. Preserves the strongest activation — the most prominent feature in each region.
- **Average pooling**: Takes the mean value. Smoother but may lose sharp features.
- **Global average pooling** (GAP): Averages each entire feature map to a single number. Replaces fully connected layers at the end of modern architectures, reducing parameters and overfitting.

A 2x2 max pool with stride 2 halves both spatial dimensions, reducing the feature map size by 4x while keeping the strongest signals.

## CNN model

```python
class ConvolutionalNetwork(nn.Module):
    def __init__(self, im_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=8*7*7, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))        # (1,28,28) -> (4,28,28)
        x = F.max_pool2d(x, 2, 2)        # (4,28,28) -> (4,14,14)
        x = F.relu(self.conv2(x))         # (4,14,14) -> (8,14,14)
        x = F.max_pool2d(x, 2, 2)         # (8,14,14) -> (8,7,7)
        x = x.view(-1, 8*7*7)            # flatten to (392,)
        x = F.relu(self.fc1(x))           # (392,) -> (32,)
        x = self.fc2(x)                   # (32,) -> (10,)
        return x
```
<!--code-toggle-->
```pseudocode
CLASS ConvolutionalNetwork:
    INIT(in_channels=1, n_classes=10):
        conv1 = CONV2D(1, 4, kernel=3, padding=1)
        conv2 = CONV2D(4, 8, kernel=3, padding=1)
        fc1   = LINEAR(8*7*7, 32)
        fc2   = LINEAR(32, 10)

    FORWARD(x):                        // Input: (1, 28, 28)
        x = RELU(conv1(x))            // -> (4, 28, 28)
        x = MAX_POOL_2D(x, size=2)    // -> (4, 14, 14)
        x = RELU(conv2(x))            // -> (8, 14, 14)
        x = MAX_POOL_2D(x, size=2)    // -> (8, 7, 7)
        x = FLATTEN(x)                // -> (392,)
        x = RELU(fc1(x))              // -> (32,)
        x = fc2(x)                    // -> (10,)
        RETURN x
```

## Modern architectures

The evolution of CNN architectures brought key innovations:

- **LeNet** (1998): Pioneered the conv-pool-conv-pool-fc pattern. Demonstrated CNNs for handwritten digit recognition.
- **AlexNet** (2012): Scaled to ImageNet with deeper networks, ReLU activations, and dropout. Won the ImageNet challenge by a large margin and launched the deep learning revolution.
- **VGGNet** (2014): Showed that using small 3x3 kernels stacked deeply (16-19 layers) outperforms larger kernels. Two 3x3 layers have the same receptive field as one 5x5 layer but with fewer parameters.
- **ResNet** (2015): Introduced **residual connections** (skip connections) that enabled training of networks with 100+ layers.

## Residual connections

Deep networks suffer from the **degradation problem**: adding more layers can actually *increase* training error, even though the network has strictly more capacity. This is counterintuitive — a deeper network should do at least as well as a shallower one, since the extra layers could just learn the identity function. In practice, learning the identity is surprisingly hard.

**Residual connections** solve this elegantly. Instead of asking the network to learn the full transformation, we ask it to learn only the *difference* from the identity:

$$
\mathbf{h}_{l+1} = \mathbf{h}_l + F(\mathbf{h}_l; \theta_l),
$$

where $F$ is the residual function (typically two conv-BN-ReLU layers). It is like telling the network: "If you cannot figure out what to do, at least copy what was already there." If the optimal transformation is close to identity, the network only needs to learn a small residual $F \approx 0$, which is much easier than learning the full mapping from scratch.

Skip connections also improve gradient flow: the gradient can flow directly through the identity path, preventing vanishing gradients even in very deep networks.

## Receptive field

The **receptive field** of a neuron is the region of the input image that can influence its activation. It grows with depth:

- A single 3x3 conv layer has a 3x3 receptive field.
- Two stacked 3x3 layers have a 5x5 receptive field.
- Three stacked 3x3 layers have a 7x7 receptive field.
- Pooling and strided convolutions increase the receptive field more aggressively.

The **effective receptive field** is typically much smaller than the theoretical one, concentrated in the center following a Gaussian distribution. Dilated convolutions and attention mechanisms can expand it efficiently.

## Transfer learning

Training a CNN from scratch on a small dataset often leads to overfitting. **Transfer learning** leverages features learned on large datasets (typically ImageNet):

- **Feature extraction**: Freeze the pre-trained convolutional layers and train only a new classification head. Works well when the target domain is similar to ImageNet.
- **Fine-tuning**: Unfreeze some or all layers and train with a small learning rate. Allows the network to adapt its features to the new domain while retaining general knowledge.

Transfer learning is one of the most impactful practical techniques in deep learning, enabling high performance even with limited labeled data. The lower layers (edges, textures) transfer well across almost any image domain; only the higher layers need task-specific adaptation.

## Evaluation

The CNN achieves approximately 97.5% accuracy on the MNIST test set, with a network about 1/8th the size of the fully connected MLP (which achieved 95.5%). This demonstrates the power of exploiting spatial structure through convolutions — fewer parameters, better results.

## Further reading

- He, K., et al. (2016). *Deep Residual Learning for Image Recognition*. The ResNet paper that made 100+ layer networks trainable.
- CS231n Stanford, *Convolutional Neural Networks for Visual Recognition*. Comprehensive course notes with excellent visualizations of feature hierarchies and receptive fields.
