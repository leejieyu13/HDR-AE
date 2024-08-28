import os
import os.path
import scipy.io as sio
import pickle
import glob

import numpy as np
import re
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import random
import datatransforms
from torchvision import transforms
import cv2
import ipdb


class TorchDataset(torch.utils.data.Dataset):
    """ Dataset for use in pytorch"""

    def __init__(self, data_dir, mat_file, opts, transform=None):
        """ Args:
                targets (string): Path to file containing optimal images
                transform (optional, callable): Optional transform to be applied on a sample
        """
        self.mat_file = mat_file
        self.dataset = sio.loadmat(mat_file)
        self.opts = opts
        self.transform = transform
        self.sequence = opts.sequence
        self.image_info = list(self.dataset.values())[3:]
        self.data_dir = data_dir
        

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        TT = transforms.ToTensor()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        bpp = 12
        img_seq = []
        EV_seq = []
      
        for i in range(0,len(self.image_info[idx]),2):                
            img_name = self.image_info[idx][i].rstrip()       
            hdr = np.load(os.path.join(self.data_dir, img_name)).astype(np.float32)
            hdr = cv2.resize(hdr, (224, 224))   
            img_seq.append(hdr)
            EV_seq.append(eval(self.image_info[idx][i+1]))
        # EV0 = [EV_seq[0], EV_seq[1]]
        sample = {'img_seq': img_seq, 'target': EV_seq}

        if self.transform:
            sample = self.transform(sample)
        
        
        return sample