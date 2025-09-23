import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure inputs are log-probabilities for numerical stability
        # If inputs are logits, apply log_softmax
        if not (inputs.is_floating_point() and inputs.min() >= -1e-8 and inputs.max() <= 1e-8):
            inputs = F.log_softmax(inputs, dim=1)

        # Calculate cross entropy loss
        # NLLLoss expects log-probabilities and integer targets
        ce_loss = F.nll_loss(inputs, targets, reduction='none')

        # Convert log_softmax to probabilities for pt calculation
        # pt is the probability of the true class
        pt = torch.exp(-ce_loss)

        # Calculate Focal Loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
