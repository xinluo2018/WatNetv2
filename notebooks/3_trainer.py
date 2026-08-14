'''
author: xin luo
create: 2026.8.7
des: training script for deep learning model
'''

import sys 
sys.path.append('/home/xin/Developer-luo/WatNetv2')  ## add the current working directory to sys.path for module import
import time
import torch
import random
import numpy as np 
import pandas as pd 
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from notebooks import config
import matplotlib.pyplot as plt
from utils.utils import read_scenes
from utils.data_loader import SceneArraySet, PatchPathSet
from utils.data_aug import GaussianNoise
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassJaccardIndex, BinaryAccuracy
from model import unet, deeplabv3plus_mobilev2, swin_unet, watnet
from model import unet_timm, unet_swin_timm

## 1. params 
patch_size = 512        ## patch size setting
patch_resize = None     ## patch resize setting
learning_rate = 1e-4     
batch_size = 8   
device = torch.device('cuda:0')  
model_name = 'watnet'  ## model name for saving
print('model:', model_name)
### traset
paths_scene_tra, paths_truth_tra = config.paths_tra_scene, config.paths_tra_truth
print(f'train_scenes: {len(paths_scene_tra)}')
## valset
paths_valset = sorted(glob(f'data/dset/val_patch/patch_{patch_size}/*'))  ## for model prediction 
print(f'vali_patch_{patch_size}: {len(paths_valset)}')

### 2. Read data
scenes_arr, truths_arr = read_scenes(paths_scene_tra, paths_truth_tra) 
scene_truth_ls =  list(zip(scenes_arr, truths_arr))
scene_truth_ls = [np.concatenate([scene, truth[:, :, np.newaxis]], axis=-1) 
                                for scene, truth in scene_truth_ls] 

## 3. dataloader
transforms_tra = v2.Compose([
      v2.ToImage(),
      v2.RandomCrop(size=(patch_size, patch_size)),
      v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.5),      ## type: ignore
      GaussianNoise(mean=0, sigma_max=0.1, p=0.5),    
      v2.RandomHorizontalFlip(p=0.5),
      v2.RandomVerticalFlip(p=0.5),
       ])
transforms_val = v2.Compose([
      v2.ToDtype(torch.float32),
       ])

tra_dset = SceneArraySet(scene_truth_list = scene_truth_ls, transforms = transforms_tra)
val_dset = PatchPathSet(paths_valset=paths_valset, transforms=transforms_val)
### Create DataLoader
tra_loader = torch.utils.data.DataLoader(tra_dset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_dset, batch_size=batch_size, num_workers=4) 

## 4. model, loss and optimizer
### 4.1 create model
# model = unet(num_bands=6)
model = watnet(num_bands=6, num_classes=2)
# model = deeplabv3plus_mobilev2(num_bands=6)
## model = swin_unet(img_size=512, num_bands=6, window_size=8)
## model = unet_timm(num_bands=6, 
##                    backbone_name='efficientnet_b0', 
##                   # backbone_name='mobilenetv3_large_100',
##                   #  backbone_name='resnet50',
##                    pretrained=True)
## model = unet_swin_timm(num_bands=6, 
##                          img_size = 512, 
##                          backbone_name='swinv2_base_window8_256',
##                          pretrained=True) 

### 4.2 create loss and optimizer  
bce_loss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

## 5. train and val loops
'''------train loops------'''
def train_loops(model, loss_fn, 
                    optimizer, 
                    tra_loader, 
                    val_loader,  
                    epoches, 
                    device, 
                    lr_scheduler=None):
    loss_tra_loops, miou_tra_loops, oa_tra_loops = [], [], []
    loss_val_loops, miou_val_loops, oa_val_loops = [], [], []
    model = model.to(device)
    size_tra_loader = len(tra_loader)
    size_val_loader = len(val_loader)
    best_miou = 0.90
    epoches_i = []
    for epoch in range(epoches):
        start = time.time()
        loss_tra, loss_val = 0, 0
        '''-----train the model-----'''
        oa_tra = BinaryAccuracy().to(device)
        miou_tra = MulticlassJaccardIndex(num_classes=2, average="macro").to(device)
        model.train()   # training mode for dropout and batchnorm
        for x_batch, y_batch in tra_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            pred = (F.sigmoid(pred) > 0.5).float()
            miou_tra.update(pred, y_batch.long())
            oa_tra.update(pred, y_batch.long())
            loss_tra += loss.item()
        miou_tra_global = miou_tra.compute()
        oa_tra_global = oa_tra.compute()
        loss_tra_global = loss_tra/size_tra_loader
        miou_tra.reset(); oa_tra.reset()

        '''----- validation the model: time consuming -----'''
        oa_val = BinaryAccuracy().to(device)
        miou_val = MulticlassJaccardIndex(num_classes=2, average="macro").to(device)

        model.eval()
        # if epoch > 500 and (epoch+1) % 2 == 0: 
        if (epoch+1) % 2 == 0: 
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                with torch.no_grad():
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)
                pred = (F.sigmoid(pred) > 0.5).float()
                miou_val.update(pred, y_batch.long())
                oa_val.update(pred, y_batch.long())
                loss_val += loss.item()
            miou_val_global = miou_val.compute()
            oa_val_global = oa_val.compute()
            loss_val_global = loss_val/size_val_loader
            miou_val.reset(); oa_val.reset()
            loss_tra_loops.append(loss_tra_global); miou_tra_loops.append(miou_tra_global.item()); oa_tra_loops.append(oa_tra_global.item())
            loss_val_loops.append(loss_val_global); miou_val_loops.append(miou_val_global.item()); oa_val_loops.append(oa_val_global.item())
            epoches_i.append(epoch)
            print(f'Ep{epoch}: tra-> Loss:{loss_tra_global:.3f},Oa:{oa_tra_global:.3f},Miou:{miou_tra_global:.3f}, '
                    f'val-> Loss:{loss_val_global:.3f},Oa:{oa_val_global:.3f}, Miou:{miou_val_global:.3f},time:{time.time()-start:.1f}s')
            ## save the best model
            if miou_val_global.item() > best_miou:
                best_miou = miou_val_global.item()       ## update best miou
                torch.save(model.state_dict(), f'model/trained/{model_name}_0{str(round(best_miou*10000))}.pth')
        else: 
            print(f'Ep{epoch}: tra-> Loss:{loss_tra_global:.3f},Oa:{oa_tra_global:.3f},Miou:{miou_tra_global:.3f}, \
                                time:{time.time()-start:.1f}s')
        if lr_scheduler:
          lr_scheduler.step(miou_tra_global)    ## if use lr_scheduler like ReduceLROnPlateau

    metrics = {'epoch':epoches_i, 'tra_loss':loss_tra_loops, 'tra_oa': oa_tra_loops, 'tra_miou': miou_tra_loops,
                'val_loss': loss_val_loops, 'val_oa': oa_val_loops, 'val_miou': miou_val_loops}
    return metrics 

if __name__ == '__main__':
    metrics = train_loops(model=model,  
                    epoches=500,   
                    loss_fn=bce_loss,   
                    optimizer=optimizer,  
                    tra_loader=tra_loader,   
                    val_loader=val_loader,   
                    # lr_scheduler=lr_scheduler,   
                    device=device)

    # torch.save(model.state_dict(), f'model/trained/{model_name}_.pth')
    ## metrics saving
    path_metrics = f'model/trained/{model_name}_training_metrics.csv'
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(path_metrics, index=False, sep=',')  


