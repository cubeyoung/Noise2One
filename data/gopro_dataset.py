import numpy as np
import os
from data.base_dataset import BaseDataset
from .imlib import imlib
from multiprocessing.dummy import Pool
from tqdm import tqdm
from util.util import augment
import random
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

# GoPro dataset
class GoProDataset(BaseDataset):
	def __init__(self, opt, split='train'):
		super(GoProDataset, self).__init__(opt)

		if self.root == '':
			rootlist = ['/mnt/ssd3/Imagedenoising/GOPRO_Large/']
			for root in rootlist:
				if os.path.isdir(root):
					self.root = root
					break

		self.batch_size = opt.batch_size
		self.patch_size = 256
		self.mode = 'RGB'  #, Y or L=
		self.imio = imlib(self.mode, lib='cv2')
		self.names, self.blur_dirs, self.gt_dirs = self._get_image_dir(self.root, split)
		self.transform = transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])
        # # Add RandAugment with N, M(hyperparameter)
		self.transform.transforms.insert(0, RandAugment(3, 9))  
	
		if split == 'train':
			self._getitem = self._getitem_train
			self.len_data = 500 * 16 # 500 * self.batch_size
		elif split == 'val':
			self._getitem = self._getitem_test
			self.len_data = len(self.names)
		elif split == 'test': 
			self._getitem = self._getitem_test
			self.len_data = len(self.names)
		else:
			raise ValueError

		self.blur_images = [0] * len(self.names)
		self.gt_images = [0] * len(self.names)
		read_images(self)

		
	def __getitem__(self, index):
		return self._getitem(index)

	def __len__(self):
		return self.len_data

	def _getitem_train(self, idx):
		idx = idx % len(self.names)

		blur_img = self.blur_images[idx]
		gt_img = self.gt_images[idx]

		blur_img, gt_img = self._crop_patch(blur_img, gt_img)
		blur_img, gt_img = augment(blur_img, gt_img)
		blur_img = np.float32(blur_img) / 255 
		gt_img = np.float32(gt_img) / 255

		hr_coord, hr_rgb = to_pixel_samples(torch.from_numpy(gt_img).contiguous())
		cell = torch.ones_like(hr_coord)
		cell[:, 0] *= 2 / gt_img.shape[-2]
		cell[:, 1] *= 2 / gt_img.shape[-1]

		return {'gt_noise': gt_img,
				'blur_img': blur_img,
				'gt_img': gt_img,
				'fname': self.names[idx], 'coord': hr_coord, 'cell': cell , 'gt': hr_rgb}

	def _getitem_test(self, idx):

		blur_img = self.blur_images[idx]
		gt_img = self.gt_images[idx]

		blur_img = np.float32(blur_img) / 255 
		gt_img = np.float32(gt_img) / 255

		# noise_root = self.gt_dirs[idx].replace('sharp', 'npy')
		# noise_root = noise_root.replace('test', 'test_noise_' + self.opt.noisetype)
		# noise_file = noise_root[:-3] + 'npy'
		# gt_noise = np.float32(np.load(noise_file, allow_pickle=True))
		hr_coord, hr_rgb = to_pixel_samples(torch.from_numpy(gt_img).contiguous())
		cell = torch.ones_like(hr_coord)
		cell[:, 0] *= 2 / gt_img.shape[-2]
		cell[:, 1] *= 2 / gt_img.shape[-1]

		return {'gt_noise': gt_img,
				'blur_img': blur_img,
				'gt_img': gt_img,
				'fname': self.names[idx], 'coord': hr_coord, 'cell': cell , 'gt': hr_rgb}

	def _crop_patch(self, blur, gt):
		ih, iw = blur.shape[-2:]
		p = self.patch_size
		pw = random.randrange(0, iw - p + 1)
		ph = random.randrange(0, ih - p + 1)
		return blur[..., ph:ph+p, pw:pw+p], \
			   gt[..., ph:ph+p, pw:pw+p]

	def _get_image_dir(self, dataroot, split=None):
		blur_dirs = []
		gt_dirs = [] 
		image_names = []

		if split == 'train' or split == 'test':
			for scene_file in os.listdir(dataroot + split +  '/'): 
				for image_file in os.listdir(dataroot + split +  '/' + scene_file + '/sharp/'):  
					image_names.append(scene_file + '-' + image_file)
					blur_dirs.append(dataroot + split +  '/' + scene_file + '/blur_gamma/' + image_file)
					gt_dirs.append(dataroot + split +  '/' + scene_file + '/sharp/' + image_file)
		elif split == 'val':
			for scene_file in os.listdir(dataroot + 'test/'):
				for image_file in os.listdir(dataroot + 'test/' + scene_file + '/sharp/'):  
					image_names.append(scene_file + '-' + image_file)
					blur_dirs.append(dataroot + 'test/' + scene_file + '/blur_gamma/' + image_file)
					gt_dirs.append(dataroot + 'test/' + scene_file + '/sharp/' + image_file)
					break
		else:
			raise ValueError

		image_names = sorted(image_names)
		blur_dirs = sorted(blur_dirs)
		gt_dirs = sorted(gt_dirs)
		
		return image_names, blur_dirs, gt_dirs


def iter_obj(num, objs):
	for i in range(num):
		yield (i, objs)

def imreader(arg):
	i, obj = arg
	for _ in range(3):
		try:
			obj.blur_images[i] = obj.imio.read(obj.blur_dirs[i])
			obj.gt_images[i] = obj.imio.read(obj.gt_dirs[i])
			failed = False
			break
		except:
			failed = True
	if failed: print('%s fails!' % obj.names[i])

def read_images(obj):
	# may use `from multiprocessing import Pool` instead, but less efficient and
	# NOTE: `multiprocessing.Pool` will duplicate given object for each process.
	print('Starting to load images via multiple imreaders')
	pool = Pool() # use all threads by default
	for _ in tqdm(pool.imap(imreader, iter_obj(len(obj.names), obj)), total=len(obj.names)):
		pass
	pool.close()
	pool.join()

if __name__ == '__main__':
	pass
