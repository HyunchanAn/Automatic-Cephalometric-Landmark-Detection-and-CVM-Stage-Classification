import torch
import torch.nn as nn
from torchvision import models

from config import NUM_LANDMARKS

class UNetHeatmapModel(nn.Module):
    def __init__(self, num_landmarks=NUM_LANDMARKS, pretrained=True):
        super(UNetHeatmapModel, self).__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # --- Encoder (ResNet-50 Backbone) ---
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1 # Output: 256 channels, 1/4 size
        self.layer2 = resnet.layer2 # Output: 512 channels, 1/8 size
        self.layer3 = resnet.layer3 # Output: 1024 channels, 1/16 size
        self.layer4 = resnet.layer4 # Output: 2048 channels, 1/32 size

        # --- Decoder (Upsampling with Skip Connections) ---
        self.up_conv4 = self._upsample_block(2048, 1024)
        self.up_conv3 = self._upsample_block(1024 + 1024, 512)
        self.up_conv2 = self._upsample_block(512 + 512, 256)

        # Final layer to produce the heatmaps
        self.final_layer = nn.Conv2d(256, num_landmarks, kernel_size=1, stride=1, padding=0)

        self.freeze_backbone()

    def _upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- Encoder --- 
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # --- Decoder with Skip Connections ---
        u4 = self.up_conv4(x4)
        u3 = self.up_conv3(torch.cat([u4, x3], 1))
        u2 = self.up_conv2(torch.cat([u3, x2], 1))

        heatmaps = self.final_layer(u2)
        
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

# --- Previous HeatmapModel (for loading old checkpoints) ---
class HeatmapModel(nn.Module):
    def __init__(self, num_landmarks=NUM_LANDMARKS, pretrained=True):
        super(HeatmapModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.upsampling_head = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.final_layer = nn.Conv2d(64, num_landmarks, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        features = self.backbone(x)
        upsampled_features = self.upsampling_head(features)
        heatmaps = self.final_layer(upsampled_features)
        return heatmaps