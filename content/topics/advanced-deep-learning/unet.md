# U-Net Architecture

## The pixel-level challenge

Suppose you're a doctor looking at a brain scan, and you need to outline exactly where a tumor is -- not just "there's a tumor in this image," but the precise boundary of every single pixel that belongs to the tumor. That's **image segmentation**: assigning a class label to every pixel, producing a mask with the same spatial dimensions as the input.

The challenge is you need two things simultaneously. You need the **big picture** -- enough context to know this blob is a tumor, not normal tissue. And you need **fine detail** -- the exact pixel boundaries where the tumor ends and healthy tissue begins. How do you build a network that sees both the forest and the trees? You build a U-shaped pipe.

## Encoder-decoder architecture

Segmentation creates a fundamental tension:

* **Downsampling** (pooling, strided convolutions) builds a large receptive field and captures high-level semantics, but it loses spatial resolution. Each pooling layer throws away positional information.
* **Upsampling** recovers spatial resolution, but the fine details lost during downsampling are gone.

An **encoder-decoder** addresses this by first compressing through an encoder (like a classification CNN) and then expanding through a decoder that restores spatial dimensions. But by the time information reaches the bottleneck, the fine details have been squeezed out.

## Skip connections: the key insight

Here's the innovation. **U-Net** adds **skip connections** that directly connect encoder layers to their corresponding decoder layers at each resolution level. Instead of forcing all information through the narrow bottleneck, skip connections give the decoder a shortcut to the fine-grained details from the encoder.

At each level, encoder feature maps are concatenated with decoder feature maps. The decoder gets both the high-level "what" from the bottleneck and the low-level "where" from the encoder -- and learns to combine them.

It's like trying to describe the exact shape of a cloud to a friend over the phone. By the time the description passes through the bottleneck of language, the fine details are lost. Skip connections are like also sending a photograph -- your friend gets both the verbal description (global context) and the actual visual details.

## The U-shape

The U-Net gets its name from its symmetric architecture. A typical configuration for 128x128 input:

**Encoder (contracting path)**:
1. 128x128 x 1 channel $\to$ 128x128 x 64 (two 3x3 convolutions)
2. Pool $\to$ 64x64 x 128
3. Pool $\to$ 32x32 x 256
4. Pool $\to$ 16x16 x 512 (bottleneck)

**Decoder (expanding path)**:
5. Upsample $\to$ 32x32, concat with encoder level 3 $\to$ 256
6. Upsample $\to$ 64x64, concat with encoder level 2 $\to$ 128
7. Upsample $\to$ 128x128, concat with encoder level 1 $\to$ 64
8. Final 1x1 convolution $\to$ 128x128 x $C$ classes

The left side compresses, the bottom is the bottleneck, and the right side expands back to full resolution. The horizontal connections across the U are the skip connections.

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

Pixel accuracy is a terrible lie when the tumor is 1% of the image. A network that predicts "not tumor" everywhere scores 99% accuracy while being completely useless. Dice loss simply asks: "How much do my blobs overlap with the real ones?" And that's all you need:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i}{\sum_i p_i + \sum_i g_i}.
$$

Dice handles class imbalance naturally because it measures the *ratio* of overlap, not the absolute count. In practice, a weighted combination of cross-entropy and Dice loss often works best.

## Big Ideas

* The encoder-decoder pattern is a general principle: compress to understand *what* something is, then expand to reconstruct *where* it is.
* Skip connections bypass the destruction that pooling inflicts on fine spatial detail -- the decoder gets a direct copy of what the encoder saw.
* Pixel accuracy is misleading with class imbalance -- Dice loss forces the model to actually find the rare class.

## What Comes Next

The U-Net is a *discriminative* model -- given an input, it predicts a structured output. But all architectures so far have been spatial: convolutions, pooling, local receptive fields. The next lesson introduces **transformers and attention mechanisms**, which throw away locality entirely. Instead of sliding a small window across the input, every position attends directly to every other position. Attention turned out to be so powerful that it displaced not just recurrence but, in many domains, convolutions too.
