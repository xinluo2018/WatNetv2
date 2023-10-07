## author: xin luo
## create: 2021.9.9
## des: simple pre-processing for the dset data(image and truth pair).

import numpy as np
from pyrsimg import readTiff, img_normalize

def read_normalize(paths_img, paths_truth, max_bands, min_bands):
    ''' des: satellite image reading and normalization
        input: 
            paths_img: list, paths of the satellite image.
            paths_truth: list, paths of the truth image.
            max_bands: int or list, the determined max values for the normalization.
            min_bands: int or list, the determined min values for the normalization.
        return:
            scenes list and truths list
    '''
    scene_list, truth_list = [],[]
    for i in range(len(paths_img)):
        ## --- data reading
        scene_ins = readTiff(paths_img[i])
        truth_ins = readTiff(paths_truth[i])
        ## --- data normalization 
        scene_arr = img_normalize(img=scene_ins.array, max_bands=max_bands, min_bands=min_bands)
        scene_arr[np.isnan(scene_arr)]=0          ### remove nan value
        scene_list.append(scene_arr), truth_list.append(truth_ins.array)
    return scene_list, truth_list

