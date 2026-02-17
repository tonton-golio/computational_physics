# U-Net Model

## Introduction

**U-Net** is a fully convolutional network for fast image segmentation. The network takes an image as input and returns a segmentation mask of the same spatial dimensions.

U-Net uses an **encoder-decoder architecture with skip connections**. Skip connections concatenate the output of an encoder layer with the corresponding decoder layer, preserving spatial information that would otherwise be lost during downsampling.

## The Model

```python
class U_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv8 = nn.Conv2d(16, 10, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.drop = nn.Dropout2d(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(x)
        # decoder
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.up(x)
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.up(x)
        x = self.softmax(x)

        return x
```
