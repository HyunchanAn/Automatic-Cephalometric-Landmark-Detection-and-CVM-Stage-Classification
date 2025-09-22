import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CVM_STAGES

class CvmOnlyNet(nn.Module):
    def __init__(self):
        super(CvmOnlyNet, self).__init__()
        # Load a pretrained ResNet-18 model
        self.backbone = models.resnet18(pretrained=True)
        
        # Get the number of input features for the classifier
        num_ftrs = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with our CVM classifier
        self.backbone.fc = nn.Linear(num_ftrs, NUM_CVM_STAGES)

    def forward(self, x):
        # The forward pass is just the backbone
        cvm_output = self.backbone(x)
        return cvm_output
