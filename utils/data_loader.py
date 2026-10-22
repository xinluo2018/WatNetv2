## author: xin luo
## create: 2026.2.11
## des: data loading throuth 3 ways: 
#       1) through pathes (particularly for the validation patches.), 
#       2) through torch.Tensor, 


import torch
import numpy as np
from torchvision import tv_tensors

class SceneArraySet(torch.utils.data.Dataset):
    '''
    des: scene and truth image reading from the np.array(): read data from memory.
    '''
    def __init__(self, scene_truth_list, transforms=None):
        '''input arrs_scene, arrs_truth are list'''
        self.scene_truth_list = scene_truth_list
        self.transforms = transforms
    def __getitem__(self, index):
        '''load images and truths'''
        scene_truth = torch.as_tensor(self.scene_truth_list[index]).permute(2, 0, 1)
        image = tv_tensors.Image(scene_truth[:-1])
        mask = tv_tensors.Mask(scene_truth[-1:])
        '''pre-processing (e.g., random crop)'''
        ### Image augmentation
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask
    def __len__(self):
        return len(self.scene_truth_list) 

class PatchPathSet(torch.utils.data.Dataset):
    def __init__(self, paths_valset, transforms=None):
        self.paths_patch_ptruth = paths_valset
        self.transforms = transforms
    def __getitem__(self, index):
        '''load patches and truths'''
        patches_truth = torch.load(self.paths_patch_ptruth[index], 
                                   weights_only=False)        
        patches = tv_tensors.Image(patches_truth[:-1].float())
        truth = tv_tensors.Mask(patches_truth[-1:].float())  
        if self.transforms is not None:
            patches, truth = self.transforms(patches, truth)
        return patches, truth
    def __len__(self):
        return len(self.paths_patch_ptruth)

