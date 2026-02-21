# Convolutional Neural Networks

## Seeing edges by hand

Before any formulas, let us do a convolution by hand. Take a tiny 5x5 image — say, a white square on a black background. Now take a small 3x3 grid of numbers called a **Sobel filter** (it looks like $[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]$). Place this little grid on the top-left corner of the image and multiply each overlapping pair of numbers, then add them all up. That gives you one number. Slide the grid one pixel to the right and repeat. Keep sliding until you have covered the whole image. What you get is a new image where bright spots mark vertical edges — places where the intensity changes sharply from left to right. That sliding-and-multiplying operation is a **convolution**, and the little grid is a **kernel**.

The key insight: a neural network can *learn* what kernels to use. Instead of hand-designing a Sobel filter, the network discovers — through backpropagation — exactly which patterns to look for.

## Why convolutions?

Fully connected networks treat each pixel independently, ignoring the spatial structure of images. **Convolutional neural networks** (CNNs) exploit three key properties:

* **Translation equivariance**: A pattern detected in one part of the image can be recognized elsewhere without learning separate weights for each position. An edge is an edge, whether it appears in the top-left corner or the bottom-right.
* **Parameter sharing**: The same kernel weights are applied at every spatial location, dramatically reducing the parameter count.
* **Locality**: Each output depends only on a small region of the input, capturing local patterns before combining them into global features.

These inductive biases make CNNs far more efficient than MLPs for image tasks. A fully connected layer connecting two 28x28 feature maps would require $784^2 \approx 600{,}000$ parameters; a 3x3 convolutional layer needs only 9.

## The convolution operation

Imagine you are sliding a little magnifying glass over the picture and asking, at every spot, "How much does this tiny 3x3 window look like the pattern I am searching for?" That sliding multiplication is convolution. Formally, a 2D convolution slides a kernel $K$ of size $k \times k$ across an input feature map $I$, computing at each position:

$$
(I * K)[i, j] = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I[i + m, j + n] \cdot K[m, n].
$$

Key parameters that control the sliding:
* **Stride**: The step size when sliding the kernel. Stride 2 halves the spatial dimensions.
* **Padding**: Adding zeros around the border. "Same" padding preserves spatial dimensions; "valid" padding shrinks them.
* **Dilation**: Inserting gaps between kernel elements to increase the receptive field without adding parameters.

[[simulation adl-convolution-demo]]

## Kernel types

Different kernels detect different things:
* **Averaging**: A uniform kernel that blurs the image by averaging nearby pixels.
* **Gaussian blur**: A weighted average with Gaussian falloff, producing smoother blurring.
* **Edge detection**: Kernels like Sobel or Laplacian that highlight intensity changes — exactly what we did by hand above.
* **Dilated (atrous) convolution**: Expands the receptive field without increasing parameters or pooling.

In a CNN, the network learns its own kernels through training. Early kernels tend to look like edge and texture detectors; deeper kernels respond to increasingly complex patterns.

## Feature hierarchies

Evolution spent millions of years wiring V1 to IT in the ventral visual stream. A modern CNN rediscovers the same strategy in a few hours of training:

* **Early layers** learn edges and color gradients — much like V1.
* **Middle layers** combine these into textures, object parts, and shapes.
* **Deep layers** recognize whole objects and scenes — much like IT cortex.

That is why transfer learning works: the lowest layers are basically universal visual primitives. Swap out the final classification head and the same edge detectors, texture filters, and part detectors serve a completely different task.

[[simulation adl-filter-evolution]]

## What if we didn't have convolutions?

Without convolutions, you would need a fully connected layer for every pixel-to-pixel connection. A single layer processing a modest 224x224 RGB image would need $224 \times 224 \times 3 \times 224 \times 224 \times 3 \approx 23$ billion parameters — for *one* layer. And the network would have to learn separately that an edge in the top-left is the same concept as an edge in the bottom-right. Convolutions give you parameter sharing and spatial awareness for free.

## Pooling layers

**Pooling** reduces spatial dimensions and provides a degree of translation invariance:

* **Max pooling**: Takes the maximum value in each window. Preserves the strongest activation — the most prominent feature in each region.
* **Average pooling**: Takes the mean value. Smoother but may lose sharp features.
* **Global average pooling** (GAP): Averages each entire feature map to a single number. Replaces fully connected layers at the end of modern architectures, reducing parameters and overfitting.

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

* **LeNet** (1998): Pioneered the conv-pool-conv-pool-fc pattern. Demonstrated CNNs for handwritten digit recognition.
* **AlexNet** (2012): Scaled to ImageNet with deeper networks, ReLU activations, and dropout. Won the ImageNet challenge by a large margin and launched the deep learning revolution.
* **VGGNet** (2014): Showed that using small 3x3 kernels stacked deeply (16-19 layers) outperforms larger kernels. Two 3x3 layers have the same receptive field as one 5x5 layer but with fewer parameters.
* **ResNet** (2015): Applied the **residual connections** from the previous lesson to CNNs, enabling training of networks with 100+ layers. ResNet demonstrated that residual connections are a natural fit for deep convolutional architectures, making the identity the default and learning only the corrections.

## Receptive field

The **receptive field** of a neuron is the region of the input image that can influence its activation. It grows with depth:

* A single 3x3 conv layer has a 3x3 receptive field.
* Two stacked 3x3 layers have a 5x5 receptive field.
* Three stacked 3x3 layers have a 7x7 receptive field.
* Pooling and strided convolutions increase the receptive field more aggressively.

The **effective receptive field** is typically much smaller than the theoretical one, concentrated in the center following a Gaussian distribution. Dilated convolutions and attention mechanisms can expand it efficiently.

[[simulation adl-receptive-field-growth]]

## Transfer learning

Training a CNN from scratch on a small dataset often leads to overfitting. **Transfer learning** leverages features learned on large datasets (typically ImageNet):

* **Feature extraction**: Freeze the pre-trained convolutional layers and train only a new classification head. Works well when the target domain is similar to ImageNet.
* **Fine-tuning**: Unfreeze some or all layers and train with a small learning rate. Allows the network to adapt its features to the new domain while retaining general knowledge.

Transfer learning is one of the most impactful practical techniques in deep learning, enabling high performance even with limited labeled data. The lower layers (edges, textures) transfer well across almost any image domain; only the higher layers need task-specific adaptation.

## Evaluation

The CNN achieves approximately 97.5% accuracy on the MNIST test set, with a network about 1/8th the size of the fully connected MLP (which achieved 95.5%). This demonstrates the power of exploiting spatial structure through convolutions — fewer parameters, better results.

## Big Ideas

* Translation equivariance is the kernel of the idea: the same edge detector works in every corner of the image, so there is no reason to learn a separate one for each location.
* Stacking small 3x3 kernels achieves the same receptive field as a large kernel but with fewer parameters and more nonlinearities in between — depth is cheaper than width for capturing long-range spatial context.
* Transfer learning works because the visual hierarchy discovered on one task (edges → textures → parts → objects) is genuinely universal — the same filters that recognize dog fur recognize cat fur.

## What Comes Next

You now have a network that exploits spatial structure — but so far it only classifies whole images. What if you want to *generate* images instead? The next lesson introduces **generative adversarial networks**, where two networks compete: a forger tries to produce convincing fakes, and a detective tries to catch them. The adversarial game drives the generator to produce outputs so realistic they fool an expert classifier — a completely different approach to learning than anything we have seen so far.

## Check Your Understanding

1. A fully connected layer connecting two 224x224 RGB feature maps would have roughly 23 billion parameters. A 3x3 convolutional layer between two feature maps with 64 channels has only about 37,000. What architectural assumption makes this compression possible, and is it always valid?
2. Global average pooling replaces the large fully connected layers at the end of modern CNNs. What information is lost when you average each entire feature map to a single number, and why is this an acceptable trade-off for classification but potentially problematic for other tasks?
3. Skip connections in ResNet allow the gradient to flow directly through the identity path, bypassing the residual block. How does this help when training very deep networks, and why does it also make the identity function the easiest solution for the network to default to?

## Challenge

Receptive field size determines how much context a neuron at a given depth can see. Calculate the theoretical receptive field of a neuron in the fifth convolutional layer for three architectures: (a) five 3x3 convolutions with no pooling, (b) five 3x3 convolutions with 2x2 max pooling after each layer, and (c) five 3x3 dilated convolutions with dilation rates 1, 2, 4, 8, 16. Now design an experiment on a spatial task where the correct answer depends on long-range context (e.g., detecting a pattern that spans half the image) and test which architecture succeeds. Does theoretical receptive field size predict practical performance?

