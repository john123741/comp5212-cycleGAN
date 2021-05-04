import math
import torch
import torch.nn as nn

class NormalNLLLoss:
    def __call__(self, x: torch.Tensor, mean: torch.Tensor, ln_var: torch.Tensor) -> torch.Tensor:
        x_prec = torch.exp(-ln_var)
        x_diff = x - mean
        x_power = (x_diff * x_diff) * x_prec * -0.5
        loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power

        return torch.mean(loss)

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()

        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
            nn.InstanceNorm2d(dim, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
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

        # encoder
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_channels, filter_number, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(filter_number, affine=False, track_running_stats=False),
                 nn.ReLU(True)]

        # transformer
        for i in range(downsampling_number):
            in_channels = filter_number * (2 ** i)
            out_channels = in_channels * 2
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False),
                      nn.ReLU(True)]

        res_block_dim = filter_number * (2 ** downsampling_number)
        for i in range(res_block_number):
            model += [ResnetBlock(res_block_dim)]

        # decoder
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


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):

    def __init__(self, input_size, num_attr=0):
        super().__init__()
                
        self.down1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 32, 7, stride=1, padding=0),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )                

        self.t1 = double_conv(256, 256)
        self.t2 = double_conv(256, 256)        
        self.t3 = double_conv(256, 256)
        self.t4 = double_conv(256, 128)       

        latent_len = input_size // 8
        self.latent_len = latent_len
        self.conv2linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_len * latent_len * 128, 64 + num_attr),
        )

        self.linear2conv = nn.Sequential(
            nn.Linear(64 + num_attr, latent_len * latent_len * 128),
        )

        self.num_attr = num_attr   

        self.up1 = nn.Sequential(
            #nn.ConvTranspose2d(128 + 256, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(128 + 256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(            
            #nn.ConvTranspose2d(256 + 128, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(            
            #nn.ConvTranspose2d(128 + 64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )      
        self.up4 = nn.Sequential(            
            #nn.ConvTranspose2d(64 + 32, 64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True),
            nn.Conv2d(64 + 32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )                 
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    
    def forward(self, x):
        conv1 = self.down1(x)
        conv2 = self.down2(conv1)
        conv3 = self.down3(conv2)
        conv4 = self.down4(conv3)

        x = self.t1(conv4)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)

        latent = self.conv2linear(x)
        latent_out = latent[:, -self.num_attr:]
        x = self.linear2conv(latent)
        x = x.view(x.shape[0], 128, self.latent_len, self.latent_len)

        x = torch.cat([x, conv4], dim=1)
        deconv1 = self.up1(x)
        x = torch.cat([deconv1, conv3], dim=1)
        deconv2 = self.up2(x)
        x = torch.cat([deconv2, conv2], dim=1)
        deconv3 = self.up3(x)
        x = torch.cat([deconv3, conv1], dim=1)
        deconv4 = self.up4(x)
        out = self.out(deconv4)
        return out, latent_out
    
    def forward_with_attr(self, x, attr):
        conv1 = self.down1(x)
        conv2 = self.down2(conv1)
        conv3 = self.down3(conv2)
        conv4 = self.down4(conv3)

        x = self.t1(conv4)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)

        latent = self.conv2linear(x)
        latent[:, -self.num_attr:] = attr
        x = self.linear2conv(latent)
        x = x.view(x.shape[0], 128, self.latent_len, self.latent_len)

        x = torch.cat([x, conv4], dim=1)
        deconv1 = self.up1(x)
        x = torch.cat([deconv1, conv3], dim=1)
        deconv2 = self.up2(x)
        x = torch.cat([deconv2, conv2], dim=1)
        deconv3 = self.up3(x)
        x = torch.cat([deconv3, conv1], dim=1)
        deconv4 = self.up4(x)
        out = self.out(deconv4)

        return out      