import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.net = nn.Sequential(
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
                       nn.ReLU(True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=True),
                       nn.InstanceNorm2d(dim, affine=False, track_running_stats=False)
        )

    def forward(self, x):
        out = x + self.net(x)
        return out


class ResnetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResnetGenerator,self).__init__()

        filter_number = 64
        res_block_number = 9
        downsampling_number = 2

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, filter_number, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(filter_number, affine=False, track_running_stats=False),
                 nn.ReLU(True)]

        for i in range(downsampling_number):
            in_channels = filter_number * (2 ** i)
            out_channels = in_channels * 2
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False),
                      nn.ReLU(True)]

        res_block_dim = filter_number * (2 ** downsampling_number)
        for i in range(res_block_number):
            model += [ResnetBlock(res_block_dim)]

        for i in range(downsampling_number):
            in_channels = filter_number * (2 ** (downsampling_number - i))
            out_channels = int(in_channels / 2)
            model += [nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(filter_number, output_channels, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


