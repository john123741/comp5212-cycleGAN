import torch
import torch.nn as nn
from torch.nn import init


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_channels=3, input_shape=(256, 256)):
        super(NLayerDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),  
            nn.Conv2d(512, 1, 4, stride=1, padding=1),                      
        )

    def forward(self, x):
        return self.model(x)

