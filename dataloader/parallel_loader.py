## author: xin luo
## create: 2021.9.15
## des: parallel data (scene to patch) loading 
## note: the line missing is in the data loading.

import torch
import threading as td
from queue import Queue
from pyrsimg import crop2size
import numpy as np
import gc

def scenes2patches(scene_list, truth_list, transforms, size=(256, 256), channel_first=False):    
    '''
    des: convert scenes to patches
    input: 
        scene_list: list, consist of scenes, the scene shape is [band, row, col].
        truth_list: list, consist of truths image, the truth shape is [row, col].
        size: tuple, the size of patch.
    return:
        patch_list: list, consist of patches.
        ptruth_list: list, consist of truth patches.
    '''
    patch_list, ptruth_list = [],[]
    if channel_first: 
        scene_list = [scene.transpose(1,2,0) for scene in scene_list]  ## set channel to the end.
    zip_data = list(zip(scene_list, truth_list))
    for scene, truth in zip_data:
        truth = truth[:,:, np.newaxis]
        img_truth = np.concatenate((scene, truth), axis=-1)
        patch_ptruth = crop2size(img=img_truth, channel_first=False).toSize(size=size)
        patch, ptruth = patch_ptruth[:,:,0:-1], patch_ptruth[:,:,-1]
        patch = patch.transpose(2,0,1)   ### set the channel first
        ### Image augmentation.
        for transform in transforms:
            patch, ptruth = transform(patch, ptruth)
        ptruth = ptruth[np.newaxis, :, :]
        patch_list.append(patch), ptruth_list.append(ptruth)
    return patch_list, ptruth_list

def job_scenes2patches(q, scene_list, truth_list, transforms, channel_first=False):
    '''
    des: 
    '''
    patch_list, ptruth_list = scenes2patches(scene_list, truth_list, transforms, size=(256, 256), channel_first=channel_first)
    q.put((patch_list, ptruth_list))

def threads_read(scene_list, truth_list, transforms, num_thread=20, channel_first=False):
    '''multi-thread reading the training data 
        cooperated with the job function
    '''
    patch_lists, ptruth_lists = [], []
    threads = []
    q = Queue()
    for i in range(num_thread):
        thread = td.Thread(target=job_scenes2patches, args=(q, scene_list, truth_list, transforms, channel_first))
        threads.append(thread)
    start = [t.start() for t in threads]  ## start the thread
    join = [t.join() for t in threads]    ## wait the thread complete
    for i in range(num_thread):
        patch_list, ptruth_list = q.get()
        patch_lists += patch_list
        ptruth_lists += ptruth_list
    return patch_lists, ptruth_lists

class threads_scene_dset(torch.utils.data.Dataset):
    ''' 
    des: dataset (patch and the truth) parallel reading from RAM memory
    input: 
        patch_list, truth_list are lists (consist of torch.tensor).
        num_thread: number of threads
    '''
    def __init__(self, scene_list, truth_list, transforms, num_thread, channel_first=False):
        self.scene_list = scene_list
        self.truth_list = truth_list
        self.num_thread = num_thread
        self.channel_first = channel_first
        self.patches_list, self.ptruth_list = threads_read(scene_list, \
                                        truth_list, transforms, num_thread, channel_first=self.channel_first)
        self.transforms = transforms
    def __getitem__(self, index):
        '''load patches and truths'''
        patch = self.patches_list[index]
        truth = self.ptruth_list[index]
        ## update the patch-based dataset.
        if index == len(self.patches_list)-1:          
            del self.patches_list, self.ptruth_list
            gc.collect()
            self.patches_list, self.ptruth_list = threads_read(self.scene_list, \
                                        self.truth_list, self.transforms, self.num_thread, channel_first=self.channel_first)
        return patch, truth

    def __len__(self):
        return len(self.patches_list)

