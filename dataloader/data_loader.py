## author: xin luo
## create: 2026.2.11
## des: data loading throuth 3 ways: 
#       1) through pathes (particularly for the validation patches.), 
#       2) through torch.Tensor, 

import torch
import numpy as np
from scipy import ndimage
from torchvision import tv_tensors


def easy_pixel_mask(water_mask, pixel_buffer=20):
    """
    Generate a mask that excludes easy pixels.
    Input:  [H, W]
    Output: [H, W]
    0: Easy pixels (ignored)
    1: Hard pixels (evaluated)
    """
    water = np.asarray(water_mask, dtype=bool)
    eroded = ndimage.binary_erosion(water, iterations=1)
    boundary = water & ~eroded
    buffer = ndimage.binary_dilation(boundary, iterations=pixel_buffer)
    return buffer.astype(np.uint8)


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

class SceneArraySet_2(torch.utils.data.Dataset):
    '''
    des: scene and truth image reading from the np.array(): read data from memory.
    '''
    def __init__(self, scene_gfea_truth_list, transforms=None, noise=None):
        '''input arrs_scene, arrs_truth are list'''
        self.scene_gfea_truth_list = scene_gfea_truth_list
        self.transforms = transforms
        self.noise = noise 
    def __getitem__(self, index):
        '''load images and truths'''
        scene_gfea_truth = torch.as_tensor(self.scene_gfea_truth_list[index]).permute(2, 0, 1)
        image = tv_tensors.Image(scene_gfea_truth[:-1])
        mask = tv_tensors.Mask(scene_gfea_truth[-1:])
        '''pre-processing (e.g., random crop)'''
        ### Image augmentation
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        if self.noise is not None:
            bands = tv_tensors.Image(image[:6])
            prior = image[6:]
            image = tv_tensors.Image(torch.cat([self.noise(bands), prior], dim=0))        
        return image, mask
    def __len__(self):
        return len(self.scene_gfea_truth_list) 


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


class PatchPathSet_2(torch.utils.data.Dataset):
    def __init__(self, paths_valset, transforms=None, pixel_buffer=20):
        self.paths_patch_ptruth = paths_valset
        self.transforms = transforms
        self.pixel_buffer = pixel_buffer

    def __getitem__(self, index):
        patches_truth = torch.load(self.paths_patch_ptruth[index], weights_only=False)
        patches = tv_tensors.Image(patches_truth[:-1].float())
        truth = tv_tensors.Mask(patches_truth[-1:].float())
        if self.transforms is not None:
            patches, truth = self.transforms(patches, truth)
        # Truth: [1, H, W] -> [H, W]
        truth_arr = truth.squeeze(0).cpu().numpy()
        # Generate easy pixel mask
        mask_arr = easy_pixel_mask(truth_arr, pixel_buffer=self.pixel_buffer)
        # Mask: [H, W] -> [1, H, W]
        easy_mask = torch.from_numpy(mask_arr).unsqueeze(0).bool()
        return patches, truth, easy_mask

    def __len__(self):
        return len(self.paths_patch_ptruth)

