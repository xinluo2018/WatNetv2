'''
author: xin luo, 
created: 2026.8.12
des: configuration file
'''

# from glob import glob
from pathlib import Path

## directories/files
dir_tra_scene = 'data/dset/train/scene' 
dir_tra_truth = 'data/dset/train/truth' 
paths_tra_scene = sorted(Path(dir_tra_scene).glob('*.tif'))
paths_tra_truth = sorted(Path(dir_tra_truth).glob('*.tif'))
# print('train scene:', len(paths_tra_scene))
dir_val_scene = 'data/dset/val/scene' 
dir_val_truth = 'data/dset/val/truth' 
paths_val_scene = sorted(Path(dir_val_scene).glob('*.tif'))
paths_val_truth = sorted(Path(dir_val_truth).glob('*.tif'))
# print('val scene:', len(paths_val_scene))
dir_result = 'data/result'



