"""
author: xin luo, 
created: 2026.10.10
des: segmentation decoder. 
ref: https://github.com/mzeromiko/vmamba       
"""

import torch
from torch import nn
from torch.nn import functional as F

class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

class PyramidPoolingModule(nn.Module):
    ## Multi-scale pyramid pooling for the deepest feature level
    def __init__(self, 
                 in_channels, 
                 channels, 
                 pool_scales):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvNormAct(in_channels, channels, 1))
            for scale in pool_scales
        ])

    def forward(self, x):
        outputs = [x]
        for branch in self.branches:
            outputs.append(F.interpolate(
                branch(x), size=x.shape[-2:], mode='bilinear', align_corners=False))
        return outputs


### 1. PyramidDecoder
class PyramidDecoder(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        if not in_channels or channels < 1 or num_classes < 1:
            raise ValueError('Require feature channels and positive decoder/classes widths')
        self.projections = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in in_channels)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvNormAct(channels, 32, 3, padding=1),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        ) 
    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError('Feature count does not match decoder projections')
        x = self.projections[-1](features[-1])
        for projection, feature in reversed(list(zip(self.projections[:-1], features[:-1]))):
            x = projection(feature) + F.interpolate(
                x, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)

#### 2. UPerNetDecoder
class UPerNetDecoder(nn.Module):
    """Framework-independent UPerNet decoder for fine-to-coarse NCHW features."""
    def __init__(self, 
                    in_channels,   # List of input feature channels from the backbone  
                    channels,      # Number of channels for the decoder layers   
                    num_classes,   # Number of output segmentation classes   
                    pool_scales=(1, 2, 3, 6), 
                    dropout=0.1):
        super().__init__()
        if len(in_channels) < 2 or channels < 1 or num_classes < 1:
            raise ValueError('UPerNet requires at least two feature levels and positive widths')
        ppm_channels = max(channels // len(pool_scales), 1)  # Channels for each pyramid pooling branch
        self.ppm = PyramidPoolingModule(in_channels[-1], ppm_channels, pool_scales)
        ppm_width = in_channels[-1] + ppm_channels * len(pool_scales)
        self.ppm_bottleneck = ConvNormAct(ppm_width, channels, 3, padding=1)
        self.laterals = nn.ModuleList([
                    ConvNormAct(width, channels, 1) for width in in_channels[:-1]
                    ])
        self.fpn_convs = nn.ModuleList([
                    ConvNormAct(channels, channels, 3, padding=1)
                    for _ in in_channels[:-1]
                        ])
        self.fpn_bottleneck = ConvNormAct(channels * len(in_channels), channels, 3, padding=1)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        # self.classifier = nn.Conv2d(channels, num_classes, 1)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvNormAct(channels, 32, 3, padding=1),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        ) 


    def forward(self, features):
        if len(features) != len(self.laterals) + 1:
            raise ValueError('Feature count does not match UPerNet levels')
        laterals = [layer(feature) for layer, feature in zip(self.laterals, features[:-1])]
        laterals.append(self.ppm_bottleneck(torch.cat(self.ppm(features[-1]), dim=1)))
        for index in range(len(laterals) - 1, 0, -1):  # from deep feature to shallow feature
            ## Upsample the deeper lateral feature and add it to the current lateral feature
            laterals[index - 1] = laterals[index - 1] + F.interpolate(
                    laterals[index], size=laterals[index - 1].shape[-2:], 
                    mode='bilinear', align_corners=False)   
        fpn_outputs = [layer(feature) for layer, feature in zip(self.fpn_convs, laterals[:-1])]
        fpn_outputs.append(laterals[-1])
        output_size = fpn_outputs[0].shape[-2:]
        fpn_outputs = [fpn_outputs[0]] + [
            F.interpolate(feature, size=output_size, mode='bilinear', align_corners=False)
            for feature in fpn_outputs[1:]
        ]
        output = self.fpn_bottleneck(torch.cat(fpn_outputs, dim=1))
        output = self.head(self.dropout(output)) 
        return output

