import torch
import torch.nn as nn
from torchvision import models

from config import NUM_LANDMARKS

class HeatmapModel(nn.Module):
    def __init__(self, num_landmarks=NUM_LANDMARKS, pretrained=True):
        super(HeatmapModel, self).__init__()
        
        # Load a pre-trained ResNet-50 model
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Backbone: Use ResNet-50 up to the last convolutional block
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze backbone by default
        self.freeze_backbone()

        # Upsampling head
        self.upsampling_head = nn.Sequential(
            # Input: 2048 channels from ResNet-50
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
        
        # Final layer to produce the heatmaps
        self.final_layer = nn.Conv2d(
            in_channels=64, 
            out_channels=num_landmarks, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )

    def forward(self, x):
        # Pass input through the backbone
        features = self.backbone(x)
        
        # Upsample the features
        upsampled_features = self.upsampling_head(features)
        
        # Generate heatmaps
        heatmaps = self.final_layer(upsampled_features)
        
        return heatmaps

    def freeze_backbone(self):
        """Freezes the parameters of the ResNet backbone."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreezes the parameters of the ResNet backbone for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True