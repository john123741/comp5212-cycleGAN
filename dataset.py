import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CycleGANStandardDataset(Dataset):
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
        
        return img