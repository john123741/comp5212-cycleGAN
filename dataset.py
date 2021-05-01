import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CycleGANStandardDataset(Dataset):
    def __init__(self, path, attr_path=None, transform=None):
        self.path = path
        self.attr_path = attr_path
        self.data = []
        self.num_attr = 0
        
        # let the user overwrite the image preprocessing
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        # load the excel file for reading attributes
        self.attributes = None
        if attr_path:
            self.df_attr = pd.read_csv(attr_path)
            self.attributes = self.df_attr.columns[1:]

            for i, row in enumerate(self.df_attr.iterrows()):
                filename = row[1][0]
                if os.path.isfile(os.path.join(path, filename)):                    
                    label = torch.from_numpy(row[1][1:].to_numpy(dtype=np.float32))
                    self.num_attr = label.shape[0]
                    self.data.append((filename, label))
                else:
                    raise Exception('The following file does not exist: %s' % os.path.join(path, filename))
            print('[%s]: %d files (with label) discovered.' % (path, len(self.data)))         
        else:
            for file in os.listdir(path):
                if os.path.isfile(os.path.join(path, file)):
                    self.data.append(file)
            print('[%s]: %d files discovered.' % (path, len(self.data)))              

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.attr_path:
            img_name, label = self.data[idx]
        else:
            img_name = self.data[idx]

        img = Image.open(os.path.join(self.path, img_name))
        img = self.transform(img)

        if self.attr_path:
            return img, label
        return img

class CycleGANTestDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.data = []
        
        # let the user overwrite the image preprocessing
        if not transform:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
        
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                self.data.append(file)
        print('[%s]: %d files discovered.' % (path, len(self.data)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data[idx]
        img = Image.open(os.path.join(self.path, img_name))
        
        img = self.transform(img)
        
        return img, img_name