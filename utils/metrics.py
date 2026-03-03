## author: xin luo
## create: 2025.6.18
## modify: 2026.2.17
## des: accuracy metric of oa(overall accuracy), miou

import torch
import numpy as np

###   ---------------- numpy array-based ---------------- 
class metrics_segm:
    def __init__(self, cla_map, truth_map, 
                 class_labels=None, mean_mode=False):
        """
        cla_map: predicted segmentation map, 2D numpy array of shape (H, W) or (B, H, W)
        truth_map: ground truth map, 2D numpy array of shape matching cla_map
        """     
        self.cla_map = cla_map
        self.truth_map = truth_map
        # Ensure input shapes match
        if cla_map.shape != truth_map.shape:
            raise ValueError("Shape of cla_map and truth_map must match.")        
        # Get unique class labels if not provided
        if class_labels is None:
            self.class_labels = np.unique(np.concatenate([cla_map.flatten(), 
                                                          truth_map.flatten()]))
        else:
            self.class_labels = np.array(class_labels)
        self.mean_mode = mean_mode

    def _get_intersection_union(self, class_label):
        """
        Helper function to compute intersection and union for a specific class
        """
        intersection = np.logical_and(self.cla_map == class_label, 
                                      self.truth_map == class_label).sum()
        union = np.logical_or(self.cla_map == class_label, 
                              self.truth_map == class_label).sum()
        return intersection, union

    @property
    def oa(self):
        """
        Overall Accuracy (OA) for binary segmentation
        """
        oa = np.mean(self.cla_map == self.truth_map)
        return oa
    @property
    def iou(self, target_class_labels=None):
        """Intersection over Union (IoU) for a specific class
        """        
        if target_class_labels is None:
            target_class_labels = self.class_labels    
        iou_scores = {}
        for c in target_class_labels:
            intersection, union = self._get_intersection_union(c)
            iou_scores[f'label_{c.item()}'] = intersection / (union + 1e-7)  # avoid division by zero
        if self.mean_mode:
            iou_scores[f'labels_mean'] = np.mean(list(iou_scores.values()))
        return iou_scores
    @property
    def dice(self, target_class_labels=None):
        """Dice Coefficient for a specific class
        """        
        if target_class_labels is None:
            target_class_labels = self.class_labels    
        dice_scores = {}
        for c in target_class_labels:
            intersection, union = self._get_intersection_union(c)
            dice_scores[f'label_{c.item()}'] = (2 * intersection) / (union + intersection + 1e-7)  # avoid division by zero
        if self.mean_mode:
            dice_scores['labels_mean'] = np.mean(list(dice_scores.values()))
        return dice_scores
    @property
    def precision(self, target_class_labels=None):
        """Precision for a specific class
        """        
        if target_class_labels is None:
            target_class_labels = self.class_labels    
        precision_scores = {}
        for c in target_class_labels:
            tp = np.logical_and(self.cla_map == c, self.truth_map == c).sum()
            fp = np.logical_and(self.cla_map == c, self.truth_map != c).sum()
            precision_scores[f'label_{c.item()}'] = tp / (tp + fp + 1e-7)  # avoid division by zero
        if self.mean_mode:
            precision_scores['labels_mean'] = np.mean(list(precision_scores.values()))
        return precision_scores

    @property
    def recall(self, target_class_labels=None):
        """Recall for a specific class
        """        
        if target_class_labels is None:
            target_class_labels = self.class_labels    
        recall_scores = {}
        for c in target_class_labels:
            tp = np.logical_and(self.cla_map == c, self.truth_map == c).sum()
            fn = np.logical_and(self.cla_map != c, self.truth_map == c).sum()
            recall_scores[f'label_{c.item()}'] = tp / (tp + fn + 1e-7)  # avoid division by zero
        if self.mean_mode:
            recall_scores['labels_mean'] = np.mean(list(recall_scores.values()))
        return recall_scores

    @property
    def f1_score(self, target_class_labels=None):
        """F1 Score for a specific class
        """        
        if target_class_labels is None:
            target_class_labels = self.class_labels    
        f1_scores = {}
        for c in target_class_labels:
            tp = np.logical_and(self.cla_map == c, self.truth_map == c).sum()
            fp = np.logical_and(self.cla_map == c, self.truth_map != c).sum()
            fn = np.logical_and(self.cla_map != c, self.truth_map == c).sum()
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1_scores[f'label_{c.item()}'] = (2 * precision * recall) / (precision + recall + 1e-7)    
        if self.mean_mode:
            f1_scores['labels_mean'] = np.mean(list(f1_scores.values()))
        return f1_scores


###  ---------------- 4d pytorch tensor-based ---------------- 
def oa_binary(pred, truth, device='cpu'):
    ''' des: calculate overall accuracy (2-class classification) for each batch
        input: 
            pred: (4D tensor) 
            truth: (4D tensor)
            device: 'cpu' or 'cuda'
    '''
    pred, truth = pred.to(device), truth.to(device)
    pred_bi = torch.where(pred>0.5, 
                          torch.ones(pred.shape, device=pred.device), 
                          torch.zeros(pred.shape, device=pred.device))
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred_bi, bins=2, min=0, max=1)
    oa = area_inter/(area_pred+0.0000001)
    oa = oa.mean()
    return oa


def miou_binary(pred, truth, device='cpu'):
    ''' des: calculate miou (2-class classification) for each batch
        input: 
            pred: (4D tensor: N*C*H*W) 
            truth: (4D tensor: N*C*H*W)
            device: 'cpu' or 'cuda'
    '''
    pred, truth = pred.to(device), truth.to(device)
    pred_bi = torch.where(pred>0.5, 
                          torch.ones(pred.shape, device=pred.device),
                          torch.zeros(pred.shape, device=pred.device))
    inter = pred_bi+truth
    area_inter = torch.histc(inter.float(), bins=3, min=0, max=2)
    area_inter = area_inter[0:3:2]
    area_pred = torch.histc(pred_bi.float(), bins=2, min=0, max=1)
    area_truth = torch.histc(truth.float(), bins=2, min=0, max=1)
    area_union = area_pred + area_truth - area_inter
    iou = area_inter/(area_union+0.0000001)
    miou = iou.mean()
    return miou 

