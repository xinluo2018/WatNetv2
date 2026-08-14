'''
author: xin luo, 
created: 2026.8.12
des: configuration file
'''

from glob import glob

## directories/files
dir_tra_scene = 'data/dset/train/scene' 
dir_tra_truth = 'data/dset/train/truth' 
dir_result = 'data/result'
paths_tra_scene = sorted(glob(dir_tra_scene+'/*.tif'))
paths_tra_truth = sorted(glob(dir_tra_truth+'/*.tif'))
