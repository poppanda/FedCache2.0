import torch 
import torch.nn as nn


class UrbanNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        print(f"Create urban with {in_channels} and {num_classes}")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_net = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_feats=False):
        feats = self.feature_net(x)
        ret = self.classifier(feats)
        if return_feats:
            return ret, feats
        return ret
