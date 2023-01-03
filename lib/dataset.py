import numpy as np
from PIL import Image
import os
import torch
from torch.utils.data import Dataset
from lib.utils import *
from torchvision import transforms

def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(1, 2)
        img = img.flip(1)
    elif rot_mode == 1:
        img = img.flip(1)
        img = img.flip(2)
    elif rot_mode == 2:
        img = img.flip(1)
        img = img.transpose(1, 2)
    return img

def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(1)
    elif flip_mode == 1:
        img = img.flip(2)
    return img



class MMDataset(Dataset):
    def __init__(self,path,type,transform=None,crop_size=192):
        # fh = open(txt, 'r')
        path0=os.path.join(path,type,'train',type.split('_')[0])
        path1=os.path.join(path,type,'train',type.split('_')[1])
        pathDir = os.listdir(path0)
        imgs0 = []
        imgs1 = []
        for i in range(len(pathDir)):
            imgs0.append(path0+'/'+pathDir[i])
            imgs1.append(path1+'/'+pathDir[i])
        self.imgs0 = imgs0
        self.imgs1 = imgs1
        self.transform = transform
        self.crop_size = crop_size
        self.suffix_transform = transforms.Compose([
                                                    transforms.GaussianBlur(kernel_size=[3],sigma=[0.01,1]),
                                                    transforms.Lambda(RandomNoise)])
        # self.loader = loader

    def __getitem__(self, index):
        img0 = Image.open(self.imgs0[index])
        if img0.mode != 'RGB':
            img0 = img0.convert('RGB')
        img1 = Image.open(self.imgs1[index])
        if img1.mode != 'RGB':
            img1 = img1.convert('RGB')
       
        img0 = self.transform(img0)
        img1 = self.transform(img1)
        img0 = self.suffix_transform(img0)
        img1 = self.suffix_transform(img1)
        flip_mode = np.random.randint(0, 3)
        rot_mode = np.random.randint(0, 4)
        img0 = flip(img0,flip_mode)
        img1 = flip(img1,flip_mode)
        img0 = rot(img0,rot_mode)
        img1 = rot(img1,rot_mode)
        img0 = (img0-img0.mean(dim=[-1,-2],keepdim=True))/(img0.std(dim=[-1,-2],keepdim=True)+1e-5)
        img1 = (img1-img1.mean(dim=[-1,-2],keepdim=True))/(img1.std(dim=[-1,-2],keepdim=True)+1e-5)
        img0, img1, aflow = Random_proj(img0,img1,self.crop_size)
        return {
            'img1': torch.FloatTensor(img0),
            'img2': torch.FloatTensor(img1),
            'aflow':aflow
        }
         

    def __len__(self):
        return len(self.imgs0)