'''
author: xin luo, 
created: 2026.8.12
des: configuration file
'''

from glob import glob

## directories/files
dir_tra_scene = 'data/dset/train/scene' 
dir_tra_truth = 'data/dset/train/truth' 
paths_tra_scene = sorted(glob(dir_tra_scene+'/*.tif'))
paths_tra_truth = sorted(glob(dir_tra_truth+'/*.tif'))
# print('train scene:', len(paths_tra_scene))
dir_val_scene = 'data/dset/val/scene' 
dir_val_truth = 'data/dset/val/truth' 
paths_val_scene = sorted(glob(dir_val_scene+'/*.tif'))
paths_val_truth = sorted(glob(dir_val_truth+'/*.tif'))
# print('val scene:', len(paths_val_scene))
dir_result = 'data/result'



