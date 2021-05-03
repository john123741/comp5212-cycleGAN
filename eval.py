import os
import pickle
import argparse
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

from model import *
from dataset import *

# For imshow image in matplotlib pyplot
# input: torch tensor, output: numpy image
def numpy_convert(tensor, to_numpy=True):
    tensor = tensor.detach()
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    # clip data to a valid range
    tensor = ((tensor - tensor.min()) / (tensor.max() - tensor.min())).cpu()
    if to_numpy:
        tensor = tensor.numpy().transpose(1, 2, 0)
    return tensor

def load_checkpoint(model, filename):
    with open(filename, 'rb') as f:
        print('Loading model checkpoint: %s' % filename)
        content = pickle.load(f)
        model.netG_A.load_state_dict(content['net_G_A'])
        model.netG_B.load_state_dict(content['net_G_B'])
        model.netD_A.load_state_dict(content['net_D_A'])
        model.netD_B.load_state_dict(content['net_D_B'])
        if 'net_E' in content and model.l_encoder is not None:
            model.l_encoder.load_state_dict(content['net_E'])
        epoch = content['epoch']
        datapath = content['dataset']
    return model, epoch, datapath

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMP5212 Project - CycleGAN Eval.py')
    # Essential parameters
    parser.add_argument('datapath', default='', help='The path containing images to convert')     
    parser.add_argument('-d', '--device', default='cuda', type=str, help='cpu / cuda (default = cuda)')         
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size (default = 2)')    
    parser.add_argument('-r', '--resume', default='', type=str, help='The path to load the .pth model') 
    parser.add_argument('-o', '--output', default='generated', type=str, help='The path to save the generated images') 
    parser.add_argument('--resize', default=256, type=int, help='The dimension to resize (used in prototype only)')   
    parser.add_argument('--num_attr', default=0, type=int, help='The number of attributes (used in prototype only)')     
    args = parser.parse_args()

    # loading parameters
    datapath = args.datapath
    device = args.device
    resume = args.resume
    output_path = args.output
    num_attr = args.num_attr

    tform = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    batch_size = args.batch_size
    path_a = os.path.join('dataset', datapath, 'testA')
    path_b = os.path.join('dataset', datapath, 'testB')    
    dataA = CycleGANTestDataset(path_a, transform=tform)
    dataB = CycleGANTestDataset(path_b, transform=tform)

    dataloaderA = DataLoader(dataA, batch_size=batch_size, shuffle=True)
    dataloaderB = DataLoader(dataB, batch_size=batch_size, shuffle=True)    
    model = CycleGAN(device=device, imgsize=(args.resize, args.resize), num_attr=num_attr)
    if resume == '':
        print('You must specify the pretrained model with --resume <pth>')
        exit(1)
    model, epsilon, _ = load_checkpoint(model, resume)

    # generate output path for testA and testB
    output_path_A = os.path.join(output_path, 'testA')
    if not os.path.exists(output_path_A):
        os.makedirs(output_path_A)
    output_path_B = os.path.join(output_path, 'testB')
    if not os.path.exists(output_path_B):
        os.makedirs(output_path_B)        
    # A => B
    cnt = 0
    for i, (img, names) in enumerate(dataloaderA):
        with torch.no_grad():
            real_A = img.to(device)
            fake_B, rec_A, attr_AB, attr_ABA = model.forward_A(real_A)
            for i in range(len(names)):
                outfilepath = os.path.join(output_path_A, names[i])
                save_image(numpy_convert(fake_B[i], to_numpy=False), outfilepath)            
                cnt += 1
            print('A => B (%s): %d / %d' % (output_path_A, cnt, len(dataloaderA.dataset)), end='\r')
    print('A => B (%s): %d / %d' % (output_path_A, cnt, len(dataloaderA.dataset)))

    # B => A
    cnt = 0
    for i, (img, names) in enumerate(dataloaderB):
        with torch.no_grad():
            real_B = img.to(device)
            fake_A, rec_B, attr_BA, attr_BAB = model.forward_B(real_B)
            for i in range(len(names)):
                outfilepath = os.path.join(output_path_B, names[i])
                save_image(numpy_convert(fake_A[i], to_numpy=False), outfilepath)  
                cnt += 1
            print('B => A (%s): %d / %d' % (output_path_A, cnt, len(dataloaderB.dataset)), end='\r')
    print('B => A (%s): %d / %d' % (output_path_A, cnt, len(dataloaderB.dataset)))
