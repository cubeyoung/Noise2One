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
import albumentations as A
import torchvision.transforms.functional as TF
import albumentations.augmentations.geometric.transforms as gt

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb
    
class TuningDataset(BaseDataset):
    def __init__(self, opt,phase):
        BaseDataset.__init__(self, opt)
        self.phase = phase
        self.dir_A = os.path.join(opt.dataroot)  
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # # Add RandAugment with N, M(hyperparameter)
        self.transform.transforms.insert(0, RandAugment(3, 9))    

        # transform = [A.RandomCrop(256,256),A.HorizontalFlip(),A.VerticalFlip(),gt.Transpose()] 
        # self.transform = A.Compose(transform)
        self.sample_q = 1024
    def _get_index(self, idx):
        return idx % len(self.A_paths)
   
    def __getitem__(self, index):
        # read a image given a random integer index
        index = self._get_index(index)
        A_path = self.A_paths[index]        
        img = Image.open(A_path)
        img = self.transform(img)
        hr_coord, hr_rgb = to_pixel_samples(img.contiguous())
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img.shape[-2]
        cell[:, 1] *= 2 / img.shape[-1]

        return {'hr': img, 'coord': hr_coord, 'cell': cell , 'gt': hr_rgb, 'lr_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)*500
