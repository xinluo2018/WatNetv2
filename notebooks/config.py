'''
author: xin luo, 
created: 2026.2.10
des: configuration file
'''

from glob import glob

sat_name = {'l5', 'l7', 'l8', 'l9', 's2'}

## directories/files
dir_scene = 'data/dset/scene/scene_nor/'
dir_truth = 'data/dset/truth/truth_tif/' 
dir_result = 'data/result/'
paths_scene = sorted(glob(dir_scene+'*/*.tif'))
paths_truth = sorted(glob(dir_truth+'*/*.tif'))

## training/validation split
ids_scene = [path.split('/')[-1].split('.')[0] for path in paths_scene] 
ids_val = list(range(0, len(ids_scene), 5))    ## every 5th scene for validation
ids_tra = sorted(list(set(range(len(ids_scene))) - set(ids_val)))
ids_scene_val = [ids_scene[i] for i in ids_val]
ids_scene_tra = [ids_scene[i] for i in ids_tra]

## traset(scene)
paths_scene_tra = [paths_scene[i] for i in ids_tra]
paths_truth_tra = [paths_truth[i] for i in ids_tra] 
## valset(scene)
paths_scene_val = [paths_scene[i] for i in ids_val]
paths_truth_val = [paths_truth[i] for i in ids_val] 

### scale and offset are given from GEE platform.  
scale = {'l5': 2.75e-05, 'l7': 2.75e-05,
                  'l8': 2.75e-05, 'l9': 2.75e-05, 'S2': 0.0001} 
offset = {'l5': -0.2, 'l7': -0.2, 
                  'l8': -0.2, 'l9': -0.2, 'S2': 0} 
