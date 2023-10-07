## author: xin luo
## create: 2021.10.27
## des: batch-based accuracy evaluation.


import torch

## -- batch-based; torch-based -- ##
## ------------------------------ ##
def oa_binary(pred, truth):
    ''' des: calculate overall accuracy (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    oa = area_inter/(area_pred+0.0000001)
    oa = oa.mean()
    return oa

def miou_binary(pred, truth):
    ''' des: calculate miou (2-class classification) for each batch
        input: 
            pred(4D tensor), and truth(4D tensor)
    '''
    pred_bi = torch.where(pred>0.5, 1., 0.)   # [N,C,H,W]
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred, bins=2, min=0, max=1)
    area_truth = torch.histc(truth.float(), bins=2, min=0, max=1)
    area_union = area_pred + area_truth - area_inter
    iou = area_inter/(area_union+0.0000001)
    miou = iou.mean()
    return miou