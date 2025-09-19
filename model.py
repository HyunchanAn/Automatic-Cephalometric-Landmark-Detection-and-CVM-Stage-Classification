import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import NUM_CVM_STAGES, NUM_LANDMARKS

class AdvancedCephNet(nn.Module):
    def __init__(self):
        super(AdvancedCephNet, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.backbone = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Remove the final fully connected layer and the avgpool layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])

        # --- Classifier Head ---
        # This will need to be calculated based on the backbone and input image size
        fc_input_size = 524288

    def forward(self, x):
        # Pass input through the ResNet base
        features = self.resnet(x)
        # The features are already flattened by the adaptive pooling layer in ResNet
        
        # Get the outputs from the two heads
        landmarks = self.landmark_head(features)
        cvm_stage = self.cvm_head(features)

        return landmarks, cvm_stage

    def unfreeze(self):
        """Unfreezes the parameters of the ResNet base for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True


class CephNet(nn.Module):
    def __init__(self):
        super(CephNet, self).__init__()
        # This is a very basic CNN architecture. 
        # You can replace this with a more advanced one like ResNet, VGG, etc.
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc_input_size = 65536

        # Fully connected layers for the shared part of the network
        self.fc1 = nn.Linear(self.fc_input_size, 512)

        # Output head for landmark detection (regression)
        self.landmark_head = nn.Linear(512, 29 * 2)

        # Output head for CVM stage classification
        # 6 CVM stages
        self.cvm_head = nn.Linear(512, 6)

    def forward(self, x):
        # Convolutional part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # Flatten the feature map
        x = x.view(-1, self.fc_input_size)

        # Fully connected part
        x = F.relu(self.fc1(x))

        # Get the outputs from the two heads
        landmarks = self.landmark_head(x)
        cvm_stage = self.cvm_head(x)

        return landmarks, cvm_stage