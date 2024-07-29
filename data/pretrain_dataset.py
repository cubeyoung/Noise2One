import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import imageio as io
from PIL import Image
from torchvision.transforms import transforms
from RandAugment import RandAugment
import random
from glob import glob
    
class PretrainDataset(BaseDataset):
    def __init__(self, opt,phase):
        BaseDataset.__init__(self, opt)
        self.phase = phase
        self.dir_A = os.path.join(opt.dataroot)  
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # Add RandAugment with N, M(hyperparameter)
        self.transform.transforms.insert(0, RandAugment(16, 9))    
        self.sample_q = 1024
    def _get_index(self, idx):
        return idx % len(self.A_paths)
   
    def __getitem__(self, index):
        # read a image given a random integer index
        index = self._get_index(index)
        A_path = self.A_paths[index]        
        img = Image.open(A_path)
        img = self.transform(img)

        return {'hr': img, 'lr_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)*1000
