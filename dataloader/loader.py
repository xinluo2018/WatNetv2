## author: xin luo
## create: 2021.9.15
## des: data loading throuth 3 ways: 
#       1) through scene pathes, 
#       2) through torch.Tensor, 
#       3) through patch paths.

import torch
import numpy as np
from pyrsimg import crop2size
from utils.geotif_io import readTiff

class scene_dset(torch.utils.data.Dataset):
    '''
    des: scene and truth image reading from the np.array(): read data from memory.
    '''
    def __init__(self, scene_list, truth_list, transforms, patch_size=[256, 256], channel_first=False):
        '''input arrs_scene, arrs_truth are list'''
        if channel_first: 
            self.scene_list = [scene_tensor.transpose(1,2,0) for scene_tensor in scene_list]
        else:
            self.scene_list = scene_list
        self.truth_list = truth_list
        self.patch_size = patch_size
        self.transforms = transforms
        self.channel_first = channel_first
    def __getitem__(self, index):
        '''load images and truths'''
        scene = self.scene_list[index]
        truth = self.truth_list[index]
        '''pre-processing (e.g., random crop)'''
        truth = truth[:,:, np.newaxis]
        scene_truth = np.concatenate((scene, truth), axis=-1)
        patch_ptruth = crop2size(img=scene_truth, channel_first=False).toSize(size=self.patch_size)
        patch, ptruth = patch_ptruth[:,:,0:-1], patch_ptruth[:,:,-1]
        patch = patch.transpose(2,0,1)   ### set the channel first            
        ### Image augmentation.
        for transform in self.transforms:
            patch, ptruth = transform(patch, ptruth)
        ptruth = torch.unsqueeze(ptruth, 0)
        return patch, ptruth

    def __len__(self):
        return len(self.truth_list)


class scene_path_dset(torch.utils.data.Dataset):
    '''
    des: pair-wise image and truth image reading from the data path (read data from disk).
    '''
    def __init__(self, paths_scene, paths_truth, transforms, patch_size=[256, 256], channel_first=False):
        self.paths_scene = paths_scene
        self.paths_truth = paths_truth
        self.transforms = transforms
        self.patch_size = patch_size
        self.channel_first = channel_first
    def __getitem__(self, index):
        '''load images and truths'''
        scene_ins = readTiff(self.paths_scene[index])
        truth_ins = readTiff(self.paths_truth[index])
        truth = truth_ins.array[:, :, np.newaxis]
        scene_truth = np.concatenate((scene_ins.array, truth), axis=-1)
        '''pre-processing (e.g., random crop)'''
        patch, truth = crop2size(img=scene_truth, channel_first=self.channel_first).toSize(size=self.patch_size)
        for transform in self.transforms:
            patch, truth = transform(patch, truth)
        return patch, torch.unsqueeze(truth, 0)

    def __len__(self):
        return len(self.paths_truth)


class patch_path_dset(torch.utils.data.Dataset):
    '''sentinel-1 patch and the truth reading from data paths (in SSD)
    !!! the speed is faster than the data reading from RAM
        time record: data (750 patches) read->1.2 s model train -> 2.9 s 
    '''
    def __init__(self, paths_patch):
        self.paths_patch = paths_patch
    def __getitem__(self, index):
        '''load patches and truths'''
        patch_pair = torch.load(self.paths_patch[index])
        patch = patch_pair[0]
        truth = patch_pair[1]
        return patch, truth
    def __len__(self):
        return len(self.paths_patch)

class patch_tensor_dset(torch.utils.data.Dataset):
    '''sentinel-1 patch and the truth reading from memory (in RAM)
    !!! the speed is faster than the data reading from RAM
        time record: data (750 patches) read->0.7 s model train -> 2.9 s 
    '''
    def __init__(self, patch_pair_list):
        self.patch_pair_list = patch_pair_list
    def __getitem__(self, index):
        '''load patches and truths'''
        patch = self.patch_pair_list[index][0]
        truth = self.patch_pair_list[index][1]
        return patch, truth
    def __len__(self):
        return len(self.patch_pair_list)
