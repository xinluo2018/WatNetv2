## author: xin luo, 
## created: 2021.7.8
## modify: 2021.10.13
## des: data augmentation before model training.
#       Note: the 3-d patch is channel-first. and the truth patch is 2-d array..

import torch
import random
import numpy as np

#### ----------------------------------------------
#### patch-based augmentation
#### ----------------------------------------------

class numpy2tensor:
    '''patch-based, numpy-based
    des: pair-wise patch and ptruth from np.array to torch.Tensor
    '''
    def __call__(self, patches, ptruth):
        if isinstance(patches, list):
            patch_tensor = [torch.from_numpy(patch.copy()).float() for patch in patches]
        else:
            patch_tensor = torch.from_numpy(patches.copy()).float()
        ptruth_tensor = torch.from_numpy(ptruth.copy())
        return patch_tensor, ptruth_tensor

class rotate:
    '''patch-based, numpy-based
    des: randomly rotation with given probability
    '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        k = random.randint(1,3)
        if isinstance(patches,list):
            patches_rot = [np.rot90(patch, k, [1, 2]) for patch in patches]
        else:
            patches_rot = np.rot90(patches, k, [1, 2])
        truth_rot = np.rot90(truth, k, [0, 1])
        return patches_rot, truth_rot

class flip: 
    '''
    patch-based, numpy-based
    Des: Randomly flip with given probability 
    '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''input image, truth are np.array'''
        if random.random() > self.p:
            return patches, truth
        if random.random() < 0.5:            ## up <-> down
            if isinstance(patches, list):
                patches_flip = [np.flip(patch, 1) for patch in patches] 
            else:
                patches_flip = np.flip(patches, 1)
            truth_flip = np.flip(truth, 0)
        else:                                ## left <-> right
            if isinstance(patches, list):
                patches_flip = [np.flip(patch, 2) for patch in patches]
            else:
                patches_flip = np.flip(patches, 2)
            truth_flip = np.flip(truth, 1)
        return patches_flip, truth_flip

class noise:
    '''patch-based, numpy-based 
       !!! slower than torch-based scripts.
       des: add randomly noise with given randomly standard deviation
    '''
    def __init__(self, prob=0.5, std_min=0.001, std_max=0.1):
        self.p = prob
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        else:
            if isinstance(patches, list):
                patches_noisy = []
                for i in range(len(patches)):
                    std = random.uniform(self.std_min, self.std_max)
                    noise = np.random.normal(loc=0, scale=std, size=patches[i].shape)
                    patches_noisy.append(patches[i]+noise)
            else:
                std = random.uniform(self.std_min, self.std_max)
                noise = np.random.normal(loc=0, scale=std, size=patches.shape)
                patches_noisy = patches + noise
            return patches_noisy, truth

class colorjitter:
    '''
        patch-based, numpy-based
        des: randomly colorjitter with given probability, 
        color jitter contains bright adjust and contrast adjust.
        color jitter is performed for per band.
    '''
    def __init__(self, prob=0.5, alpha=0.1, beta=0.1):
        self.p = prob
        self.alpha = alpha
        self.beta = beta
        # self.t = t
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        if isinstance(patches, list):
            num_band = patches[0].shape[0]
            alpha, beta = [], []
            for i in range(patches[0].shape[0]):
                alpha.append(random.uniform(1-self.alpha, 1+self.alpha)) 
                beta.append(random.uniform(-self.beta, self.beta))
            for i in range(len(patches)):
                alpha += alpha
                beta += beta
            patches_cat = np.concatenate(patches, 0)
            patches_aug = []
            for i in range(patches_cat.shape[0]):
                band_aug = alpha[i]*patches_cat[i:i+1]+beta[i]
                band_aug = np.clip(band_aug, 0, 1)
                patches_aug.append(band_aug)
            patches_aug = np.concatenate(patches_aug, 0)
            patches_aug = [patches_aug[0:num_band], patches_aug[num_band:2*num_band], patches_aug[2*num_band:]]
        else: 
            patches_aug = []
            ### loop for bands
            for i in range(patches.shape[0]):                
                alpha = random.uniform(1-self.alpha, 1+self.alpha)
                beta = random.uniform(-self.beta, self.beta)
                band_aug = alpha*patches[i:i+1]+beta
                band_aug = np.clip(band_aug, 0, 1)
                patches_aug.append(band_aug)
            patches_aug = np.concatenate(patches_aug, 0)
        return patches_aug, truth


class torch_rotate:
    '''patch-based, torch-based
       des: randomly rotation with given probability
       '''
    def __init__(self, prob=0.5):
        self.p = prob
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        k = random.randint(1,3)
        if isinstance(patches,list):
            patches_rot = [torch.rot90(patch, k, [1, 2]) for patch in patches]
        else: 
            patches_rot = torch.rot90(patches, k, [1, 2])
        truth_rot = torch.rot90(truth, k, [0, 1])
        return patches_rot, truth_rot

class torch_noise:
    '''
        patch-based, torch-based (faster than numpy-based).
        des: add randomly noise with given randomly standard deviation
    '''
    def __init__(self, prob=0.5, std_min=0.001, std_max =0.1):
        self.p = prob
        self.std_min = std_min
        self.std_max = std_max
    def __call__(self, patches, truth):
        '''image, truth: torch.Tensor'''
        if random.random() > self.p:
            return patches, truth
        if isinstance(patches,list):
            patches_noisy = []
            for i in range(len(patches)):
                std = random.uniform(self.std_min, self.std_max)
                noise = torch.normal(mean=0, std=std, size=patches[i].shape)
                patches_noisy.append(patches[i].add(noise))
        else:
            std = random.uniform(self.std_min, self.std_max)
            noise = torch.normal(mean=0, std=std, size=patches.shape)
            patches_noisy = patches.add(noise)
        return patches_noisy, truth
