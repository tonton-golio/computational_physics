# U-Net Architecture

## The pixel-level challenge

Suppose you are a doctor looking at a brain scan, and you need to outline exactly where a tumor is — not just "there is a tumor in this image," but the precise boundary of every single pixel that belongs to the tumor. This is **image segmentation**: assigning a class label to every pixel in an image, producing a segmentation mask with the same spatial dimensions as the input.

The challenge is that you need two things simultaneously. You need the **big picture** — enough context to know that this blob is a tumor and not normal tissue. And you need **fine detail** — the exact pixel boundaries where the tumor ends and healthy tissue begins. How do you build a network that sees both the forest and the trees at the same time? You build a U-shaped pipe.

Applications span many domains:
* **Medical imaging**: Organ and tumor segmentation in CT/MRI scans.
* **Autonomous driving**: Identifying roads, pedestrians, vehicles, and obstacles.
* **Satellite imagery**: Land use classification, building detection, deforestation monitoring.
* **Microscopy**: Cell counting and boundary detection in biological research.

## Encoder-decoder architecture

Segmentation requires both **global context** (what object is this?) and **fine spatial detail** (where exactly are its boundaries?). This creates a fundamental tension:

* **Downsampling** (pooling, strided convolutions) builds a large receptive field and captures high-level semantics, but it loses spatial resolution. Each pooling layer throws away positional information.
* **Upsampling** (transposed convolutions, bilinear interpolation) recovers spatial resolution, but the fine details that were lost during downsampling are gone.

An **encoder-decoder** architecture addresses this by first compressing the input through an encoder (like a classification CNN) and then expanding it through a decoder that progressively restores spatial dimensions. But there is a problem: by the time the information reaches the bottleneck, the fine details have been squeezed out.

## Skip connections: the key insight

The innovation of **U-Net** is **skip connections** that directly connect encoder layers to their corresponding decoder layers at each resolution level. Instead of forcing all information through the narrow bottleneck, skip connections give the decoder a shortcut to the fine-grained details from the encoder.

At each level, the encoder feature maps are concatenated with the decoder feature maps. The decoder gets both the high-level "what" from the bottleneck and the low-level "where" from the encoder — and it learns to combine them.

## What if we didn't have skip connections?

Without skip connections, the decoder would have to reconstruct fine spatial details from only the compressed bottleneck representation. It is like trying to describe the exact shape of a cloud to a friend over the phone, and then asking them to draw it. By the time the description passes through the bottleneck of language, the fine details are lost. Skip connections are like also sending your friend a photograph — they get both your verbal description (global context) and the actual visual details.

## The U-shape

The U-Net gets its name from its symmetric U-shaped architecture. A typical configuration for 128x128 input:

**Encoder (contracting path)**:
1. 128x128 x 1 channel $\to$ 128x128 x 64 (two 3x3 convolutions)
2. Pool $\to$ 64x64 x 64 $\to$ 64x64 x 128 (two 3x3 convolutions)
3. Pool $\to$ 32x32 x 128 $\to$ 32x32 x 256 (two 3x3 convolutions)
4. Pool $\to$ 16x16 x 256 $\to$ 16x16 x 512 (bottleneck)

**Decoder (expanding path)**:
5. Upsample $\to$ 32x32 x 512 $\to$ concat with encoder level 3 $\to$ 32x32 x 256
6. Upsample $\to$ 64x64 x 256 $\to$ concat with encoder level 2 $\to$ 64x64 x 128
7. Upsample $\to$ 128x128 x 128 $\to$ concat with encoder level 1 $\to$ 128x128 x 64
8. Final 1x1 convolution $\to$ 128x128 x $C$ (number of classes)

The left side of the U compresses, the bottom is the bottleneck, and the right side expands back to full resolution. The horizontal connections across the U are the skip connections.

[[simulation adl-unet-architecture]]

## U-Net model

```python
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=10):
        super().__init__()
        # Encoder
        self.enc1 = self._block(in_ch, 16)
        self.enc2 = self._block(16, 32)
        self.enc3 = self._block(32, 64)
        self.bottleneck = self._block(64, 128)
        self.pool = nn.MaxPool2d(2, 2)
        # Decoder
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self._block(128, 64)   # 64+64 from skip
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self._block(64, 32)    # 32+32 from skip
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self._block(32, 16)    # 16+16 from skip
        self.out_conv = nn.Conv2d(16, out_ch, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
```
<!--code-toggle-->
```pseudocode
CLASS UNet:
    INIT(in_ch=1, out_ch=10):
        // Encoder blocks: each has two 3x3 convolutions + ReLU
        enc1 = CONV_BLOCK(in_ch, 16)
        enc2 = CONV_BLOCK(16, 32)
        enc3 = CONV_BLOCK(32, 64)
        bottleneck = CONV_BLOCK(64, 128)
        pool = MAX_POOL_2D(size=2)
        // Decoder: upsample + conv blocks
        up3 = TRANSPOSED_CONV2D(128, 64, size=2, stride=2)
        dec3 = CONV_BLOCK(128, 64)    // 64+64 from skip
        up2 = TRANSPOSED_CONV2D(64, 32, size=2, stride=2)
        dec2 = CONV_BLOCK(64, 32)     // 32+32 from skip
        up1 = TRANSPOSED_CONV2D(32, 16, size=2, stride=2)
        dec1 = CONV_BLOCK(32, 16)     // 16+16 from skip
        out_conv = CONV2D(16, out_ch, kernel=1)

    FORWARD(x):
        // Encoder
        e1 = enc1(x)
        e2 = enc2(pool(e1))
        e3 = enc3(pool(e2))
        b  = bottleneck(pool(e3))
        // Decoder with skip connections (concatenate)
        d3 = dec3(CONCAT(up3(b), e3))
        d2 = dec2(CONCAT(up2(d3), e2))
        d1 = dec1(CONCAT(up1(d2), e1))
        RETURN out_conv(d1)
```

## Loss functions for segmentation

Why not just use cross-entropy? Standard cross-entropy treats each pixel independently, which can be disastrous with class imbalance. If a tumor occupies 1% of the image, a network that predicts "not tumor" everywhere achieves 99% pixel accuracy while being completely useless. Specialized loss functions address this:

* **Cross-entropy**: $\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i} \sum_{c} y_{ic} \log \hat{p}_{ic}$. Simple but biased toward majority classes.

* **Dice loss**: Directly optimizes the Dice coefficient, which measures overlap between prediction and ground truth:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i},
$$

where $p_i$ and $g_i$ are predicted and ground truth values at pixel $i$. Dice loss handles class imbalance naturally because it measures the *ratio* of overlap, not the absolute count.

* **IoU (Jaccard) loss**: Similar to Dice but based on intersection over union: $\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$.

In practice, a weighted combination of cross-entropy and Dice loss often works best.

## Variants

* **U-Net++**: Introduces nested skip connections with dense blocks between encoder and decoder, allowing the model to capture features at multiple semantic scales.
* **Attention U-Net**: Adds attention gates to skip connections, letting the network learn which spatial regions and features are most relevant for the task. Not all encoder features are equally useful; attention helps the decoder focus on what matters.
* **3D U-Net**: Extends the architecture to volumetric data using 3D convolutions, essential for medical imaging (CT scans, MRI volumes).

## Practical considerations

* **Data augmentation for segmentation**: Standard augmentations (rotation, flipping, elastic deformation) must be applied identically to both the input image and the segmentation mask. If you rotate the image, you must rotate the mask by the same angle.
* **Class imbalance**: In medical imaging, the region of interest (e.g., tumor) may occupy less than 1% of the image. Use Dice loss, focal loss, or class-weighted cross-entropy.
* **Post-processing**: Connected component analysis and conditional random fields (CRFs) can refine segmentation boundaries after prediction.

## Big Ideas

* The encoder-decoder pattern is a general principle: compress to understand what something is, then expand to reconstruct where it is — the bottleneck forces global reasoning, and the decoder recovers local precision.
* Skip connections are a shortcut for details: by the time information reaches the bottleneck, fine spatial structure has been destroyed by pooling; skip connections bypass the destruction and hand the decoder a direct copy.
* Pixel accuracy is a misleading metric when classes are imbalanced — a model that labels every pixel as "background" can score 99% accuracy while being completely useless; Dice loss forces the model to actually find the rare class.
* The U-shape is not just a clever name: the symmetric structure means every level of abstraction in the encoder has a corresponding level of reconstruction in the decoder, with a direct bridge between them.

## What Comes Next

The U-Net is a *discriminative* model — given an input, it predicts a structured output. But all the architectures so far have been spatial: convolutions, pooling, local receptive fields. The next lesson introduces **transformers and attention mechanisms**, which throw away locality entirely. Instead of sliding a small window across the input, every position attends directly to every other position. Attention turned out to be so powerful that it eventually displaced not just recurrence but, in many domains, convolutions too.

## Check Your Understanding

1. A standard encoder-decoder without skip connections is asked to segment a 256x256 image after compressing it to a 4x4 bottleneck. The bottleneck has 512 channels. Does the decoder have enough information to recover precise pixel boundaries? What information is irretrievably lost?
2. Dice loss measures the ratio of overlap between prediction and ground truth. For a perfectly empty prediction mask (all pixels predicted as background), what is the Dice loss? For a perfectly full prediction mask (all pixels predicted as foreground)? What does this tell you about the loss landscape near trivial solutions?
3. When applying data augmentation during segmentation training, why is it essential to apply the same random transformation to both the image and the mask? What would happen to the training signal if you applied different random crops to each?

## Challenge

Attention U-Net adds attention gates on the skip connections, letting the decoder selectively weight which spatial locations in the encoder features are most useful. Design an experiment to test whether this actually helps, and if so, in what regime. Specifically: train a standard U-Net and an Attention U-Net on a segmentation task, then visualize the attention weights at each skip connection level. Do the attention maps look interpretable? Do they activate on the object boundaries? Does the benefit of attention grow or shrink as the class imbalance worsens?

