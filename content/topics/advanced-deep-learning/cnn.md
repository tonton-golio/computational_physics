# Convolutional Neural Networks

## Seeing edges by hand

Before any formulas, let's do a convolution by hand. Take a tiny 5x5 image -- say, a white square on a black background. Now take a small 3x3 grid of numbers called a **Sobel filter**: $[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]$. Place this little grid on the top-left corner of the image, multiply each overlapping pair of numbers, add them up. That gives you one number. Slide the grid one pixel to the right and repeat. What you get is a new image where bright spots mark vertical edges -- places where the intensity changes sharply. That sliding-and-multiplying operation is a **convolution**, and the little grid is a **kernel**.

And here's the key insight: a neural network can *learn* what kernels to use. Instead of hand-designing a Sobel filter, the network discovers -- through backpropagation -- exactly which patterns to look for.

## Why convolutions?

Fully connected networks treat each pixel independently, ignoring spatial structure. **Convolutional neural networks** (CNNs) exploit three properties that make them absurdly more efficient:

* **Translation equivariance**: A pattern detected in one part of the image can be recognized elsewhere. An edge is an edge, whether it's in the top-left or bottom-right.
* **Parameter sharing**: The same kernel weights apply at every spatial location, dramatically slashing the parameter count.
* **Locality**: Each output depends only on a small region, capturing local patterns before combining them into global features.

A fully connected layer connecting two 28x28 feature maps needs $784^2 \approx 600{,}000$ parameters. A 3x3 convolutional layer needs only 9.

## The convolution operation

Imagine sliding a tiny magnifying glass over a picture, asking at every spot: "How much does this 3x3 window look like the pattern I'm searching for?" That sliding multiplication is convolution. Formally:

$$
(I * K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i + m, j + n] \cdot K[m, n].
$$

Key parameters:
* **Stride**: The step size when sliding. Stride 2 halves the spatial dimensions.
* **Padding**: Adding zeros around the border to preserve dimensions.
* **Dilation**: Inserting gaps between kernel elements to increase the receptive field without adding parameters.

## Feature hierarchies

Evolution spent millions of years wiring V1 to IT in the visual cortex. A modern CNN rediscovers the same strategy in a few hours of training:

* **Early layers** learn edges and color gradients -- much like V1.
* **Middle layers** combine these into textures, object parts, and shapes.
* **Deep layers** recognize whole objects and scenes -- much like IT cortex.

That's why transfer learning works: the lowest layers are basically universal visual primitives. Swap out the final classification head and the same edge detectors serve a completely different task.

## Pooling layers

**Pooling** reduces spatial dimensions and provides translation invariance:

* **Max pooling**: Takes the maximum in each window -- preserves the strongest feature in each region.
* **Average pooling**: Takes the mean. Smoother but may lose sharp features.
* **Global average pooling** (GAP): Averages each entire feature map to a single number. Replaces fully connected layers at the end of modern architectures, reducing parameters and overfitting.

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

LeNet proved the idea. AlexNet made the world notice. ResNet showed that depth plus skip connections is unstoppable. That's the whole arc in three sentences.

## Receptive field

The **receptive field** of a neuron is the region of the input image that can influence its activation. It grows with depth:

* One 3x3 conv layer: 3x3 receptive field.
* Two stacked: 5x5.
* Three stacked: 7x7.
* Pooling and strided convolutions expand it more aggressively.

The **effective receptive field** is typically much smaller than the theoretical one, concentrated in the center following a Gaussian distribution.

[[simulation adl-receptive-field-growth]]

## Transfer learning

Training a CNN from scratch on a small dataset often overfits. **Transfer learning** leverages features learned on large datasets like ImageNet:

* **Feature extraction**: Freeze pre-trained convolutional layers, train only a new classification head.
* **Fine-tuning**: Unfreeze some or all layers, train with a small learning rate to adapt features while retaining general knowledge.

The lower layers (edges, textures) transfer well across almost any image domain. Only the higher layers need task-specific adaptation. This is one of the most impactful practical techniques in deep learning.

## Big Ideas

* Translation equivariance means the same edge detector works in every corner of the image -- no reason to learn a separate one for each location.
* Stacking small 3x3 kernels achieves the same receptive field as a large kernel but with fewer parameters and more nonlinearities -- depth is cheaper than width for spatial context.
* Transfer learning works because the visual hierarchy is genuinely universal -- the same filters that recognize dog fur recognize cat fur.

## What Comes Next

You now have a network that exploits spatial structure -- but so far it only classifies whole images. What if you want to *generate* images instead? The next lesson introduces **generative adversarial networks**, where two networks compete: a forger tries to produce convincing fakes, and a detective tries to catch them. The adversarial game drives the generator to produce outputs so realistic they fool an expert classifier.
