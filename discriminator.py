import torch
import torch.nn as nn
from torch.nn import init


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_channels, input_size, num_attr=0):
        super(NLayerDiscriminator, self).__init__()
        self.num_attr = num_attr
        self.input_size = input_size
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),                        
        )

        self.d_out = nn.Conv2d(512, 1, 4, stride=1, padding=1)        
        if num_attr > 0:
            final_size = (self.input_size // 8 - 1) // 2
            self.q_out = nn.Sequential(
                nn.Conv2d(512, 32, 4, stride=2, padding=1),
                nn.Flatten(),
                nn.Linear(final_size*final_size*32, num_attr),
            )

    def forward(self, x):
        x = self.backbone(x)
        return self.d_out(x)

    def q_forward(self, x):
        x = self.backbone(x)
        return self.q_out(x)
         

