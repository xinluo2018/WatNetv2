## Data directory structure
```
Data/
├── dset/                
│   ├── scene
│   │   ├── scene_ori   ## original data
│   │   │   ├──l5 
│   │   │   ├──l7 
│   │   │   ├──l8 
│   │   │   ├──l9 
│   │   │   ├──s2 
│   │   ├── scene_nor  ## nomalized data      
│   │   │   ├──l5 
│   │   │   ├──l7 
│   │   │   ├──l8 
│   │   │   ├──l9 
│   │   │   ├──s2 
│   └── truth
│   │   ├──truth_gpkg
│   │   ├──truth_tif
│   └── valset        ## validation set splitted from whole dataset    
│       │── patch_512
│       │── patch_1024
├── result

```

