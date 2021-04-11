import torch
import torch.nn as nn
from torch.nn import init


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(NLayerDiscriminator, self).__init__()
        filter_number = 64
        layer_number = 3

        sequence = [nn.Conv2d(input_channels, filter_number, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]
        coefficient = 1
        last_coefficient = 1
        for n in range(1, layer_number):
            last_coefficient = coefficient
            coefficient = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(last_coefficient * filter_number, coefficient * filter_number, kernel_size=4, stride=2, padding=1, bias=True),
                nn.InstanceNorm2d(coefficient * filter_number, affine=False, track_running_stats=False),
                nn.LeakyReLU(0.2, True)
            ]
        last_coefficient = coefficient
        coefficient = min(2 ** n, 8)

        sequence += [
            nn.Conv2d(last_coefficient * filter_number, coefficient * filter_number, kernel_size=4, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(coefficient * filter_number, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(coefficient * filter_number, 1, kernel_size=4, stride=1, padding=1)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

