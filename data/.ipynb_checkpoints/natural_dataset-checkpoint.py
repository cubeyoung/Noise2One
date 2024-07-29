import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
import imageio as io
import random
from glob import glob
class NaturalDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt,phase):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.phase = phase
        self.dir_A = os.path.join(opt.dataroot_valid)  
        path_tif = glob(opt.dataroot_valid)
        self.path_high_dose_train = []
        self.path_low_dose_train = []
        
        for p in path_tif:
            tmp_path_full_dose = glob(p + '/*.tif')
            tmp_path_low_dose = glob(p + '/*.tif')
            
            self.path_high_dose_train.extend(tmp_path_full_dose)
            self.path_low_dose_train.extend(tmp_path_low_dose) 
        print(self.path_low_dose_train)
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
    def patch(self, A):
        A,B = self.get_patch(
                    A,B,
                    patch_size=128,
                    scale=1,
                    multi=False,
                    input_large=False
                )
        #img = common.augment(img)
        return A,B
    
    def get_patch(self,*args, patch_size=96, scale=2, multi=False, input_large=False):
        ih, iw = args[0].shape[:2]

        if not input_large:
            p = scale if multi else 1
            tp = p * patch_size
            ip = tp // scale
        else:
            tp = patch_size
            ip = patch_size

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)

        if not input_large:
            tx, ty = scale * ix, scale * iy
        else:
            tx, ty = ix, iy

        ret = [
            args[0][iy:iy + ip, ix:ix + ip,:],
            *[a[ty:ty + tp, tx:tx + tp,:] for a in args[1:]]
        ]

        return ret  
    def augment(self,*args, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5
        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)
            return img

        return [_augment(a) for a in args] 
    
    def np2Tensor(self,*args):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()

            return tensor

        return [_np2Tensor(a) for a in args]
    def normalize(self,x):
        min = x.min()
        max = x.max()
        x = (x - x.min())/(x.max()-x.min())
        return x,min,max          
        
    def _get_index(self, idx):
        if self.phase == 'train':
            return idx % len(self.A_paths)
        else:
            return idx    
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        index = self._get_index(index)
        low_path = self.path_low_dose_train[index]
        full_path = self.path_high_dose_train[index]
        low = np.expand_dims(io.imread(low_path),2)
        high = np.expand_dims(io.imread(full_path),2)        
        low,min,max = self.normalize(low)
        high,_,_ = self.normalize(high)
        low= self.np2Tensor(low)[0]
        high= self.np2Tensor(high)[0]
        return {'lr': low, 'hr': high, 'lr_paths': low_path, 'max': max, 'min':min}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.phase == 'train':
            return len(self.path_low_dose_train)
        elif self.phase == 'test':
            return len(self.path_low_dose_train)
        else:
            return len(self.path_low_dose_train)