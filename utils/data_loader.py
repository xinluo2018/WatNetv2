## author: xin luo
## create: 2026.2.11
## des: data loading throuth 3 ways: 
#       1) through pathes (particularly for the validation patches.), 
#       2) through torch.Tensor, 


import torch
import numpy as np

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
        scene_truth = self.scene_truth_list[index]
        '''pre-processing (e.g., random crop)'''
        ### Image augmentation
        if self.transforms is not None:
            scene_truth = self.transforms(scene_truth)
        patches, ptruth = scene_truth[0:-1], scene_truth[-1:]
        return patches, ptruth
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
        if self.transforms is not None:
            patches_truth = self.transforms(patches_truth)
        patches = patches_truth[0:-1]
        truth = patches_truth[-1:]
        return patches, truth
    def __len__(self):
        return len(self.paths_patch_ptruth)

