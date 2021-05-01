import os
import torch
import pickle
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
    parser.add_argument('--attr1', default='', type=str, help='The file of the attributes labels (for A)')
    parser.add_argument('--attr2', default='', type=str, help='The file of the attributes labels (for B)')
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
    attr_path_A = args.attr1
    attr_path_B = args.attr2
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
    assert (not attr_path_A and not attr_path_B) or (attr_path_A and attr_path_B), "For representation learning, both --attr1 and --attr2 must be filled. Only one found."
    dataA = CycleGANStandardDataset(path_a, attr_path=attr_path_A, transform=downscaler)
    dataB = CycleGANStandardDataset(path_b, attr_path=attr_path_B, transform=downscaler)
    num_attr = dataA.num_attr
    assert dataA.num_attr == dataB.num_attr, "The two attribute lists A (%d) and B (%d) have different lengths" % (dataA.num_attr, dataB.num_attr)    

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
        model, last_epoch, _ = load_checkpoint(model, resume)
        print('Resume from epoch: %d' % last_epoch)

    total_iter_number = 0

    visualize_path = 'generated'
    total_epoch = fixed_learning_rate_epoches + linearly_decay_learning_rate_epoches
    for epoch in range(last_epoch, total_epoch):
        model.update_learning_rate()
        for i, data in enumerate(dataloaderA):  # inner loop within one epoch
            if attr_path_A and attr_path_B:
                imgA, labelA = data
                real_A = imgA.to(device)
                labelA = labelA.to(device)
                imgB, labelB = next(iter(dataloaderB))
                real_B = imgB.to(device)
                labelB = labelB.to(device)
            else:
                real_A = data.to(device)
                real_B = next(iter(dataloaderB)).to(device)
                labelA, labelB = None, None
            
            # forward (compute fake image and reconstruction images)
            fake_B, rec_A, fake_A, rec_B, attr_AB, attr_ABA, attr_BA, attr_BAB = model.forward(real_A, real_B)
            # G_A and G_B
            model.set_requires_grad([model.netD_A, model.netD_B], False)  # Ds require no gradients when optimizing Gs
            model.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            loss_G_list = model.backward_G(real_A, real_B, fake_A, fake_B, rec_A, rec_B, attr_AB, attr_ABA, attr_BA, attr_BAB, labelA, labelB)
            loss_G, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_attr = loss_G_list
            model.optimizer_G.step()       # update G_A and G_B's weights
            # D_A and D_B
            model.set_requires_grad([model.netD_A, model.netD_B], True)
            model.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
            loss_D_A, loss_D_A_real, loss_D_A_fake = model.backward_D_A(real_B, fake_B)      # calculate gradients for D_A
            loss_D_B, loss_D_B_real, loss_D_B_fake = model.backward_D_B(real_A, fake_A)      # calculate graidents for D_B
            model.optimizer_D.step()  # update D_A and D_B's weights
                        
            # log progress
            if total_iter_number % print_loss_freq == 0:
                print("[Epoch %d] Loss G (GA, GB, CycleA, CycleB): %f, %f, %f, %f Loss D (Ar, Af, Br, Bf): %f, %f, %f, %f Loss Attr: %f" % \
                    (epoch, loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_D_A_real, loss_D_A_fake, loss_D_B_real, loss_D_B_fake, loss_attr))
                # compute accuracy                
                if attr_path_A and attr_path_B:                    
                    with torch.no_grad():
                        bin_attr_AB = torch.where(attr_AB >= 0.5, 1, 0)
                        bin_attr_BA = torch.where(attr_BA >= 0.5, 1, 0)
                        acc_AB = (labelA == bin_attr_AB).sum() / torch.numel(labelA)
                        acc_BA = (labelB == bin_attr_BA).sum() / torch.numel(labelB)
                    print('Attributes Classification Accuracy - A->B: %f, B->A: %f' % (acc_AB.item(), acc_BA.item()))

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
                'net_E': model.l_encoder.state_dict() if model.l_encoder else None,
            }
            with open(output_path, 'wb') as f:
                pickle.dump(dump_content, f)
                print('Model checkpoint saved as: %s' % output_path)
            


