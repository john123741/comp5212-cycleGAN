import os
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt

from discriminator import *
from generator import *
from loss import *
from model import *
from dataset import *
from eval import load_checkpoint, numpy_convert

from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='COMP5212 Project - CycleGAN Train.py')
    # Essential parameters
    parser.add_argument('datapath', default='apple2orange', help='The path of the dataset') 
    parser.add_argument('-a', '--num_attr', default=0, type=int, help='The number of continuous attributes to represent (default = 0)')
    parser.add_argument('-d', '--device', default='cuda', type=str, help='cpu / cuda (default = cuda)')     
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size (default = 2)')    
    parser.add_argument('--f_epoch', default=100, type=int, help='Number of epochs for fixed learning rate (default = 100)') 
    parser.add_argument('--d_epoch', default=100, type=int, help='Number of epochs for decaying learning rate (default = 100)') 
    parser.add_argument('--save', default=20, type=int, help='The frequency (in epoch) of saving the model (default = 20)')
    parser.add_argument('--print', default=10, type=int, help='The frequency (in iter) of printing loss (default = 10)') 
    parser.add_argument('--generate', default=100, type=int, help='The frequency (in iter) of saving generated images (default = 100)') 
    parser.add_argument('-r', '--resume', default='', type=str, help='The path to load the .pth model')     
    # Trivial parameters for debugging
    parser.add_argument('--resize', default=64, type=int, help='The dimension to resize (used in prototype only)') 
    args = parser.parse_args()

    # loading parameters
    num_attr = args.num_attr
    device = args.device
    datapath = args.datapath
    resume = args.resume
    path_a = os.path.join('dataset', datapath, 'trainA')
    path_b = os.path.join('dataset', datapath, 'trainB')

    downscaler = transforms.Compose([
        transforms.Resize((args.resize, args.resize)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])
    dataA = CycleGANStandardDataset(path_a, transform=downscaler)
    dataB = CycleGANStandardDataset(path_b, transform=downscaler)   

    batch_size = args.batch_size
    fixed_learning_rate_epoches = args.f_epoch
    linearly_decay_learning_rate_epoches = args.d_epoch
    save_model_freq = args.save
    print_loss_freq = args.print
    save_generated_img = args.generate    

    dataloaderA = DataLoader(dataA, batch_size=batch_size, shuffle=True)
    dataloaderB = DataLoader(dataB, batch_size=batch_size, shuffle=True)
    last_epoch = 0
    model = CycleGAN(device=device, imgsize=(args.resize, args.resize), num_attr=num_attr)
    if resume != '':
        model, last_epoch, _, num_attr_loaded = load_checkpoint(model, resume, device=device)
        print('Resume from epoch: %d' % last_epoch)
        if num_attr_loaded != num_attr:
            print('Warning: loaded num_attr (%d) is different to targeted num_attr (%d)' % (num_attr_loaded, num_attr))

    total_iter_number = 0

    visualize_path = 'generated'
    total_epoch = fixed_learning_rate_epoches + linearly_decay_learning_rate_epoches
    for epoch in range(last_epoch, total_epoch):
        model.update_learning_rate()
        for i, data in enumerate(dataloaderA):  # inner loop within one epoch
            real_A = data.to(device)
            real_B = next(iter(dataloaderB)).to(device)
            
            # forward (compute fake image and reconstruction images)
            fake_B, rec_A, fake_A, rec_B, attr_AB, attr_ABA, attr_BA, attr_BAB = model.forward(real_A, real_B)
            # G_A and G_B
            model.set_requires_grad([model.netD_A, model.netD_B], False)  # Ds require no gradients when optimizing Gs
            model.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            loss_G_list = model.backward_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
            loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B = loss_G_list
            model.optimizer_G.step()       # update G_A and G_B's weights
            # D_A and D_B
            model.set_requires_grad([model.netD_A, model.netD_B], True)
            model.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            loss_D_A, loss_D_A_real, loss_D_A_fake = model.backward_D_A(real_B, fake_B)      # calculate gradients for D_A
            loss_D_B, loss_D_B_real, loss_D_B_fake = model.backward_D_B(real_A, fake_A)      # calculate graidents for D_B            
            # Q
            loss_attr = 0
            if num_attr > 0:
                loss_attr = model.backward_Q(fake_A.detach(), fake_B.detach(), attr_AB.detach(), attr_BA.detach()) 
            # update D_A and D_B's weights
            model.optimizer_D.step()                    
                        
            # log progress
            if total_iter_number % print_loss_freq == 0:
                print("[Epoch %d] Loss G (GA, GB, CycleA, CycleB): %f, %f, %f, %f Loss D (Ar, Af, Br, Bf): %f, %f, %f, %f Loss Attr: %f" % \
                    (epoch, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_D_A_real, loss_D_A_fake, loss_D_B_real, loss_D_B_fake, loss_attr))

            # save generated image
            if total_iter_number % save_generated_img == 0:
                fig = plt.figure()
                sp = plt.subplot(2, 3, 1)
                sp.set_title("A")
                sp.set_axis_off()
                plt.imshow(numpy_convert(real_A))
                sp = plt.subplot(2, 3, 2)
                sp.set_title("A => B")
                sp.set_axis_off()
                plt.imshow(numpy_convert(fake_B))
                sp = plt.subplot(2, 3, 3)
                sp.set_title("A => B => A")
                sp.set_axis_off()
                plt.imshow(numpy_convert(rec_A))
                sp = plt.subplot(2, 3, 4)
                sp.set_title("B")
                sp.set_axis_off()
                plt.imshow(numpy_convert(real_B))
                sp = plt.subplot(2, 3, 5)
                sp.set_title("B => A")
                sp.set_axis_off()
                plt.imshow(numpy_convert(fake_A))
                sp = plt.subplot(2, 3, 6)
                sp.set_title("B => A => B")
                sp.set_axis_off()
                plt.imshow(numpy_convert(rec_B))                
                plt.savefig(os.path.join(visualize_path, 'epoch{}_iter{}.png'.format(epoch, total_iter_number)), bbox_inches='tight')   
                plt.close(fig)
            total_iter_number += 1

        # save model checkpoint
        if epoch % save_model_freq == 0 or epoch == total_epoch - 1:
            output_path = os.path.join('saved', '{}_epoch_{}.pth'.format(datapath, epoch))
            dump_content = {
                'epoch': epoch,
                'dataset': datapath,
                'net_G_A': model.netG_A.state_dict(),
                'net_G_B': model.netG_B.state_dict(),
                'net_D_A': model.netD_A.state_dict(),
                'net_D_B': model.netD_B.state_dict(),
                'num_attr': num_attr,
            }
            torch.save(dump_content, output_path)        
            print('Model checkpoint saved as: %s' % output_path)
            


