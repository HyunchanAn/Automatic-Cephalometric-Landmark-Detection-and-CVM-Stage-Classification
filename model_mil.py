
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class AttentionMIL(nn.Module):
    def __init__(self, num_classes=6, backbone_pretrained=True):
        super(AttentionMIL, self).__init__()
        self.num_classes = num_classes

        # 1. 백본 모델 (ResNet-18)
        resnet = models.resnet18(pretrained=backbone_pretrained)
        # 마지막 분류 레이어를 제외하고 특징 추출기로 사용
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # ResNet-18의 출력 특징 차원은 512
        feature_dim = 512

        # 2. 어텐션 네트워크
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 3. 최종 분류기 (Ordinal Regression을 위한 CORAL 방식)
        # K개의 클래스에 대해 K-1개의 로짓을 출력
        self.classifier = nn.Linear(feature_dim, num_classes - 1)

    def forward(self, x):
        # x shape: (batch_size, num_patches, C, H, W)
        batch_size, num_patches, C, H, W = x.shape
        
        # 모든 패치를 하나의 배치로 취급하여 특징 추출
        x = x.view(batch_size * num_patches, C, H, W)
        
        # (batch_size * num_patches, 512, 4, 4) for 128x128 input
        H = self.features(x)
        
        # Apply adaptive average pooling to get a fixed-size output
        # (batch_size * num_patches, 512, 1, 1)
        H = F.adaptive_avg_pool2d(H, (1, 1))

        # (batch_size * num_patches, feature_dim)
        H = H.view(batch_size * num_patches, -1)
        
        # (batch_size, num_patches, feature_dim)
        H = H.view(batch_size, num_patches, -1)

        # 어텐션 가중치 계산
        # A_unnormalized: (batch_size, num_patches, 1)
        A_unnormalized = self.attention(H)
        # A: (batch_size, num_patches, 1)
        A = F.softmax(A_unnormalized, dim=1)
        
        # 가중치와 특징을 곱하여 최종 특징 벡터 계산
        # M: (batch_size, feature_dim)
        M = torch.sum(A * H, dim=1)

        # 최종 분류
        logits = self.classifier(M)
        return logits


