from generator import ResnetGenerator, UNet
from discriminator import NLayerDiscriminator
from loss import GANLoss
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import itertools

def init_net(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class CycleGAN(nn.Module):
    def __init__(self, device, imgsize=(256, 256), isTrain=True):
        super(CycleGAN, self).__init__()
        self.netG_A = UNet().to(device) #ResnetGenerator(3, 3).to(device)
        self.netG_A.apply(init_net)
        self.netG_B = UNet().to(device) #ResnetGenerator(3, 3).to(device)
        self.netG_B.apply(init_net)

        if isTrain:
            self.netD_A = NLayerDiscriminator(3).to(device)
            self.netD_A.apply(init_net)
            self.netD_B = NLayerDiscriminator(3).to(device)
            self.netD_B.apply(init_net)

            #self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            #self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = GANLoss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=0.0002, betas=(0.5, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

            fixed_learning_rate_epoches = 100
            linearly_decay_learning_learning_rate_epoches = 100
            def learning_rate_lambda(epoch):
                lr_l = 1.0 - max(0, epoch - fixed_learning_rate_epoches) / float(
                    linearly_decay_learning_learning_rate_epoches + 1)
                return lr_l

            self.schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=learning_rate_lambda) for optimizer in self.optimizers]

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def forward(self, real_A, real_B):
        fake_B = self.netG_A(real_A)  # G_A(A)
        rec_A = self.netG_B(fake_B)   # G_B(G_A(A))
        fake_A = self.netG_B(real_B)  # G_B(B)
        rec_B = self.netG_A(fake_A)   # G_A(G_B(B))
        return fake_B, rec_A, fake_A, rec_B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator
        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)        
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D, loss_D_real, loss_D_fake

    def backward_D_A(self, real_B, fake_B):
        """Calculate GAN loss for discriminator D_A"""
        loss_D_A, loss_D_A_real, loss_D_A_fake = self.backward_D_basic(self.netD_A, real_B, fake_B)
        return loss_D_A, loss_D_A_real, loss_D_A_fake

    def backward_D_B(self, real_A, fake_A):
        """Calculate GAN loss for discriminator D_B"""
        loss_D_B, loss_D_B_real, loss_D_B_fake = self.backward_D_basic(self.netD_B, real_A, fake_A)
        return loss_D_B, loss_D_B_real, loss_D_B_fake

    def backward_G(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = 0.1
        lambda_A = 10
        lambda_B = 10
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.netG_A(real_B)
            loss_idt_A = self.criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.netG_B(real_A)
            loss_idt_B = self.criterionIdt(idt_B, real_A) * lambda_A * lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterionGAN(self.netD_A(fake_B), True)
        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterionGAN(self.netD_B(fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * lambda_B
        # combined loss and calculate gradients        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        return loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad        