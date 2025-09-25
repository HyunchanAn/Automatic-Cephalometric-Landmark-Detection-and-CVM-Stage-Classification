import torch
import torch.nn as nn
from torchvision import models

from config import NUM_LANDMARKS

class UNetHeatmapModel(nn.Module):
    def __init__(self, num_landmarks=NUM_LANDMARKS, pretrained=True):
        super(UNetHeatmapModel, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # --- Encoder (ResNet-50 Backbone) ---
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # Output: 256 channels
        self.layer2 = resnet.layer2 # Output: 512 channels
        self.layer3 = resnet.layer3 # Output: 1024 channels
        self.layer4 = resnet.layer4 # Output: 2048 channels

        # --- Decoder (Upsampling with Skip Connections) ---
        # Upsampling from layer4 to layer3's size
        self.up_conv4 = self._upsample_block(2048, 1024)
        # Upsampling from (up_conv4 + layer3) to layer2's size
        self.up_conv3 = self._upsample_block(1024 + 1024, 512)
        # Upsampling from (up_conv3 + layer2) to layer1's size
        self.up_conv2 = self._upsample_block(512 + 512, 256)
        # Upsampling from (up_conv2 + layer1) to layer0's size
        self.up_conv1 = self._upsample_block(256 + 256, 256)
        # Final upsampling to heatmap size
        self.up_conv0 = self._upsample_block(256, 128)

        # Final layer to produce the heatmaps
        self.final_layer = nn.Conv2d(128, num_landmarks, kernel_size=1, stride=1, padding=0)

        # Freeze backbone by default
        self.freeze_backbone()

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder --- 
        x0 = self.layer0(x)      # Size: 1/4, Channels: 64
        x1 = self.layer1(x0)     # Size: 1/4, Channels: 256
        x2 = self.layer2(x1)     # Size: 1/8, Channels: 512
        x3 = self.layer3(x2)     # Size: 1/16, Channels: 1024
        x4 = self.layer4(x3)     # Size: 1/32, Channels: 2048

        # --- Decoder with Skip Connections ---
        u4 = self.up_conv4(x4)
        u3 = self.up_conv3(torch.cat([u4, x3], 1))
        u2 = self.up_conv2(torch.cat([u3, x2], 1))
        u1 = self.up_conv1(torch.cat([u2, x1], 1))
        u0 = self.up_conv0(u1)

        # --- Final Output ---
        heatmaps = self.final_layer(u0)
        
        return heatmaps

    def freeze_backbone(self):
        """Freezes the parameters of the ResNet backbone."""
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the parameters of the ResNet backbone for fine-tuning."""
        for layer in [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = True
