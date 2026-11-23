## author: xin luo, 
## created: 2023.10.14; modify: 
## des: data augmentation for multiscale patches

import torch
import random
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F


## build custom transforms
class GaussianNoise(v2.Transform):
    def __init__(self, mean = 0.0, sigma_max=0.1, p=0.5):
        super().__init__()
        self.mean = mean
        self.sigma_max = sigma_max
        self.p = p
    def transform(self, inpt, params):  # rewrite transform function to update sigma
        patch, ptruth = inpt[0:-1], inpt[-1:]
        if torch.rand(1) < self.p:
            self.sigma = torch.rand(1)*self.sigma_max  ## update sigma        
            noise_patch = torch.randn_like(patch) * self.sigma            
            patch = patch + noise_patch
            inpt = torch.cat([patch, ptruth], dim=0)
        return inpt


class CropScales:
    def __init__(self, scales):
        self.scales = scales
    def __call__(self, img):
        patch_top = random.randint(0, img.shape[1]-self.scales[0]) 
        patch_left = random.randint(0, img.shape[2]-self.scales[0]) 
        patch = F.crop(inpt = img, top=patch_top, left=patch_left, \
                                                height=self.scales[0], width=self.scales[0])
        patch_med = F.center_crop(inpt = patch, output_size=self.scales[1])
        patch_low = F.center_crop(inpt = patch, output_size=self.scales[2])
        patch = F.resize(inpt=patch, size=self.scales[-1], antialias=True)
        patch_med = F.resize(inpt=patch_med, size=self.scales[-1], antialias=True)
        return [patch[0:-1], patch_med[0:-1], patch_low[0:-1], torch.round(patch_low[-1:])]

class CropVaryScale:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        patch_row = random.randint(self.size, img.shape[1]-self.size) 
        patch_col = random.randint(self.size, img.shape[2]-self.size) 
        patch_size = min(patch_row, patch_col)
        patch_truth = v2.RandomCrop(size=patch_size)(img)
        patch_resize = F.resize(inpt=patch_truth, size=self.size, antialias=True)
        return [patch_resize[0:-1], torch.round(patch_resize[-1:])] 

class FlipScales:
    def __init__(self, prob = 0.5):
        self.prob = prob
    def __call__(self, patches_truth):
        patches_truth_flip = [] 
        if random.random() > self.prob:
          choice = random.random()
          for patch in patches_truth:
              if choice > 0.5:
                 patch_flip = F.horizontal_flip(inpt=patch)
              else: 
                 patch_flip = F.vertical_flip(inpt=patch)
              patches_truth_flip.append(patch_flip)
        else:
           patches_truth_flip = patches_truth
        return patches_truth_flip


class NoisyScales:
  def __init__(self, std=[0, 0.001]):
        self.std = std
  def __call__(self, patches_truth):
        '''patch_truth: torch.Tensor, image''' 
        patches_truth_noisy = []
        for patch in patches_truth[0:-1]:
            patch_ = patch.clone()
            std = random.uniform(self.std[0], self.std[1])
            noise = torch.normal(mean=0, std=std, size=patch_.shape, device=patch_.device)
            patch_ = patch_.add(noise)
            patches_truth_noisy.append(patch_)
        patches_truth_noisy.append(patches_truth[-1])
        return patches_truth_noisy

