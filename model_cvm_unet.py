import torch
import torch.nn as nn
from torchvision import models

from config import NUM_CVM_STAGES

class UNetCVMClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CVM_STAGES, pretrained=True):
        super(UNetCVMClassifier, self).__init__()
        
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # --- Encoder (ResNet-50 Backbone) ---
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # --- Decoder (Upsampling with Skip Connections) ---
        self.up_conv4 = self._upsample_block(2048, 1024)
        self.up_conv3 = self._upsample_block(1024 + 1024, 512)
        self.up_conv2 = self._upsample_block(512 + 512, 256)

        # --- Classification Head ---
        # Input to head is the output of the last decoder block (256 channels)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

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

        # --- Classification ---
        x = self.avgpool(u2)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        
        return logits
