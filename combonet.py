from torchvision import models
import torch.nn as nn
import torch

class ComboNet(nn.Module):
    def __init__(self, input_shape=(96, 96, 3)):
        super(ComboNet, self).__init__()
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_extractor_backbone = efficientnet.features
        num_features = 1280
        self.fen = nn.Sequential(
            feature_extractor_backbone,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        self.classifier_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    def forward(self, img1, img2):
        f1_feature = self.fen(img1)
        f2_feature = self.fen(img2)
        concatted_feature = torch.cat((f1_feature, f2_feature), dim=1)
        output = self.classifier_head(concatted_feature)
        return output