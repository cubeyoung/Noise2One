import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import imageio as io
import random
from PIL import Image
from torchvision.transforms import transforms
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

class ValidDataset(BaseDataset):
    def __init__(self, opt,phase):
        BaseDataset.__init__(self, opt)
        self.phase = phase
        self.dir_img = os.path.join(opt.dataroot_valid)  
        self.img_paths = sorted(make_dataset(self.dir_img, opt.max_dataset_size))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])           
    def _get_index(self, idx):
        if self.phase == 'train':
            return idx % len(self.img_paths)
        else:
            return idx    
    def __getitem__(self, index):
        index = self._get_index(index)
        img_path = self.img_paths[index]   
        img = Image.open(img_path)
        img = self.transform(img)
        hr_coord, hr_rgb = to_pixel_samples(img.contiguous())
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img.shape[-2]
        cell[:, 1] *= 2 / img.shape[-1]        
        return {'hr': img, 'coord': hr_coord, 'cell': cell , 'gt': hr_rgb, 'lr_paths': img_path}


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_paths)