from generator import ResnetGenerator
from discriminator import NLayerDiscriminator
from loss import GANLoss
from torch.nn import init
from torch.optim import lr_scheduler
import torch
import itertools

def init_net(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)

class CycleGAN:
    def __init__(self):
        self.netG_A = ResnetGenerator(3, 3)
        self.netG_A.apply(init_net)
        self.netG_B = ResnetGenerator(3, 3)
        self.netG_B.apply(init_net)

        if self.isTrain:
            self.netD_A = NLayerDiscriminator(3)
            self.netD_A.apply(init_net)
            self.netD_B = NLayerDiscriminator(3)
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
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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

    def set_input(self, data):
        self.real_A = data['A']
        self.real_B = data['B']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))

    def optimize_parameters(self):
        pass