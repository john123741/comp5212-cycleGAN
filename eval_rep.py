import os
import pickle
import argparse
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image

from eval import numpy_convert, load_checkpoint
from model import *
from dataset import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COMP5212 Project - CycleGAN Eval_rep.py')
    # Essential parameters
    parser.add_argument('datapath', default='', help='The image file to load (a single image)')
    parser.add_argument('type', default='A', type=str, help='The class of input image (A/B)')    
    parser.add_argument('-r', '--resume', default='', type=str, help='The path to load the .pth model') 
    parser.add_argument('-l', '--latent', default='', type=str, help='The file storing the attributes in tsv. Switch to decode mode if this is non-empty.')

    parser.add_argument('-d', '--device', default='cpu', type=str, help='cpu / cuda (default = cpu)')   

    parser.add_argument('--resize', default=256, type=int, help='The dimension to resize (used in prototype only)')
    parser.add_argument('--num_attr', default=12, type=int, help='The number of attributes (used in prototype only)')       
    args = parser.parse_args()

    # loading parameters
    datapath = args.datapath
    input_type = args.type
    assert input_type == 'A' or input_type == 'B', "Non-existing input type: %s. Only [A/B] allowed." % input_type
    latent = args.latent
    device = args.device
    resume = args.resume    
    resize = args.resize
    num_attr = args.num_attr

    tform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5), (.5,.5,.5)),
    ])

    if os.path.isfile(datapath):
        img = Image.open(datapath)        
        img = tform(img)
    else:
        print('Image file not found: %s' % datapath)
        exit(1)

    model = CycleGAN(device=device, imgsize=(resize, resize), num_attr=num_attr)
    assert resume != '', "You must specify the pretrained model with --resume <pth>"
    model, epsilon, _ = load_checkpoint(model, resume)


    # hard-code so far
    hardcode_attr = ['Bangs','Blond_Hair','Brown_Hair','Eyeglasses','Mouth_Slightly_Open','Narrow_Eyes','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings',
        'Wearing_Hat','Wearing_Necklace']

    if latent == '': # No latent => perform attribute classification
        with torch.no_grad():
            img = img.unsqueeze(0).to(device)
            if input_type == 'A':
                fake, _, attr, _ = model.forward_A(img)
            else: # B
                fake, _, attr, _ = model.forward_B(img)
            # print out the attributes
            attr = attr.squeeze(0)
            print('Attr:')
            for i in range(len(hardcode_attr)):
                print('%s: %f' % (hardcode_attr[i], attr[i].item()))
            plt.imshow(numpy_convert(fake.squeeze(0)))
            plt.show()
    else: # latent exists => modify latent and generate new image
        assert os.path.isfile(latent), "CSV file not found: %s" % os.path.isfile(latent)
        df = pd.read_csv(latent)
        attr = torch.from_numpy(df.iloc[0].to_numpy(dtype=np.float32))
        with torch.no_grad():
            img = img.unsqueeze(0).to(device)
            if input_type == 'A':
                fake, attr = model.forward_A_with_attr(img, attr)
            else: # B
                fake, attr = model.forward_B_with_attr(img, attr)
            # print out the attributes
            attr = attr.squeeze(0)
            print('Attr:')
            for i in range(len(hardcode_attr)):
                print('%s: %f' % (hardcode_attr[i], attr[i].item()))                     
            # plot the new generated output
            plt.imshow(numpy_convert(fake.squeeze(0)))
            plt.show()       
            




