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

# Convert the latent vector in the generator to disentangled attributes
class LatentEncoder(nn.Module):
    def __init__(self, input_size, num_attr):
        super().__init__()
        final_size = 256 * (input_size // 4) * (input_size // 4)
        self.num_attr = num_attr
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(final_size, num_attr),
            nn.Sigmoid(), # => [0, 1]
        )

        self.upsampler = nn.Upsample((input_size, input_size))
    
    def forward(self, x):
        out = self.encoder(x)
        out_map = out.unsqueeze(-1).unsqueeze(-1) # n, c, 1, 1        
        out_map = self.upsampler(out_map)
        return out, out_map

class UNet(nn.Module):

    def __init__(self, latent_encoder=None):
        super().__init__()
                
        self.down1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, stride=1, padding=0),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )        

        self.t1 = double_conv(256, 256)
        self.t2 = double_conv(256, 256)        
        self.t3 = double_conv(256, 256)
        self.t4 = double_conv(256, 256)       
        self.t5 = double_conv(256, 256)
        self.t6 = double_conv(256, 256)      

        if not (latent_encoder is None):
            latent_attr = latent_encoder.num_attr 
        else:
            latent_attr = 0

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256 + latent_attr, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(256, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(128, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64 + 128, 64, kernel_size=3, stride=1, padding=1, output_padding=0, bias=True),
            nn.InstanceNorm2d(64, affine=False, track_running_stats=False),
            nn.ReLU(True)
        )        
        self.out = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, padding=0),
            nn.Tanh()
        )

        self.latent_encoding = False
        if latent_encoder:
            self.latent_encoding = True
            self.l_encoder = latent_encoder
    
    def forward(self, x):
        conv1 = self.down1(x)
        conv2 = self.down2(conv1)
        conv3 = self.down3(conv2)

        x = self.t1(conv3)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)

        if self.latent_encoding:
            attr, attr_map = self.l_encoder(x)
            x = torch.cat([x, conv3, attr_map], dim=1)
        else:
            attr = None
            x = torch.cat([x, conv3], dim=1)

        deconv1 = self.up1(x)
        x = torch.cat([deconv1, conv2], dim=1)
        deconv2 = self.up2(x)
        x = torch.cat([deconv2, conv1], dim=1)
        deconv3 = self.up3(x)
        out = self.out(deconv3)
        return out, attr
    
    def forward_with_attr(self, x, attr):
        conv1 = self.down1(x)
        conv2 = self.down2(conv1)
        conv3 = self.down3(conv2)

        x = self.t1(conv3)
        x = self.t2(x)
        x = self.t3(x)
        x = self.t4(x)
        x = self.t5(x)
        x = self.t6(x)

        attr = attr.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        attr_map = self.l_encoder.upsampler(attr)

        x = torch.cat([x, conv3, attr_map], dim=1)
        deconv1 = self.up1(x)
        x = torch.cat([deconv1, conv2], dim=1)
        deconv2 = self.up2(x)
        x = torch.cat([deconv2, conv1], dim=1)
        deconv3 = self.up3(x)
        out = self.out(deconv3)
        return out, attr        