## author: xin luo
## creat: 2022.4.3, modify: 2023.2.3
## des: deeplabv3plus model with Mobilev2 backbone


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.mobilenet import MobileNetV2


def conv1x1_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv4x4_bn_relu(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, \
                            kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

####--------------for the deeplabv3_plus-----------####
def aspp_conv(in_channels, out_channels, dilation):
    '''aspp convolution: atrous_conv + bn + relu'''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, \
                            padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class aspp_pooling(nn.Module):
    '''aspp pooling: pooling + 1x1 conv + bn + relu'''
    def __init__(self, in_channels, out_channels):
        super(aspp_pooling, self).__init__()
        self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),     ## size -> 1x1
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
    def forward(self, x):
        size = x.shape[-2:]
        x = self.layers(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

class aspp(nn.Module):
    '''aspp module: consist of aspp conv, aspp pooling...'''
    def __init__(self, in_channels, atrous_rates):
        super(aspp, self).__init__()
        out_channels = 256   
        modules = []
        # branch 1：conv2d+bn+relu
        modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)  ## 
        ## branch 2-4：atrous convolution
        modules.append(aspp_conv(in_channels, out_channels, rate1))
        modules.append(aspp_conv(in_channels, out_channels, rate2))
        modules.append(aspp_conv(in_channels, out_channels, rate3))
        ## branch 5：pooling
        modules.append(aspp_pooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules) 
        ## for merged branches：conv2d+bn+relu+dropout
        self.project = nn.Sequential(
                    nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.5))
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class Mobilev2_feat(nn.Module):
    '''return:
        low-level + middle-level + high-level features'''
    def __init__(self, num_bands=4):
        super(Mobilev2_feat, self).__init__()
        self.backbone = MobileNetV2(num_bands=num_bands)
    def forward(self,x):
        x = self.backbone.head(x)
        x = self.backbone.body.inverted_0(x)
        fea_low = x    # size -> 1/2, num_channel -> 16
        x = self.backbone.body.inverted_1(x) 
        fea_mid = x    # size -> 1/4, num_channel -> 24
        x = self.backbone.body.inverted_2(x)
        x = self.backbone.body.inverted_3(x)
        fea_high = x   # size -> 1/16, num_channel -> 64
        return fea_low, fea_mid, fea_high
        
class deeplabv3plus_mobilev2(nn.Module): 
    def __init__(self, num_bands, num_classes, channels_fea=[16, 24, 64]):
        ''' 
        Improvement: 1. use mobilenetv2 as backbone model; 
                     2. use multiple level features (more a mid-level feature than deeplabv3plus).
        channels_fea (list) -> [channels_low, channels_mid, channels_high] '''

        super(deeplabv3plus_mobilev2, self).__init__()
        self.name = 'deeplabv3plus_mobilev2'
        self.aspp_channels = 256
        self.backbone = Mobilev2_feat(num_bands=num_bands)
        self.aspp = aspp(in_channels=channels_fea[2], atrous_rates=[12, 24, 36])
        self.mid_layer = conv1x1_bn_relu(channels_fea[1], 128)
        self.high_mid_layer = nn.Sequential(
                        conv1x1_bn_relu(128+self.aspp_channels, 128),
                        conv3x3_bn_relu(128, 128)
                        )
        self.low_layer = conv1x1_bn_relu(channels_fea[0], 128)
        self.high_mid_low_layer = nn.Sequential(
                        deconv4x4_bn_relu(128+128, 256),
                        nn.Dropout(0.5),
                        conv1x1_bn_relu(256, 128),
                        conv3x3_bn_relu(128, 128),
                        nn.Dropout(0.1),
                        )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
                    nn.Sigmoid())
        else: 
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128,out_channels=num_classes, kernel_size=1),
                    nn.Softmax(dim=1))

    def forward(self,x):
        ### 
        fea_low, fea_mid, fea_high = self.backbone(x)

        ### ------high level feature
        x_fea_high = self.aspp(fea_high)        # channels:256
        x_fea_high = F.interpolate(x_fea_high, \
                        fea_mid.size()[-2:], mode='bilinear', align_corners=True)
        ### ------mid-level feature, and concat
        x_fea_mid = self.mid_layer(fea_mid)
        x_fea_high_mid = torch.cat([x_fea_high, x_fea_mid], dim=1)
        x_fea_high_mid = self.high_mid_layer(x_fea_high_mid)
        x_fea_high_mid = F.interpolate(x_fea_high_mid, \
                        fea_low.size()[-2:], mode='bilinear', align_corners=True)
        ### ------low-level feature, and concat
        x_fea_low = self.low_layer(fea_low)
        x_fea_high_mid_low = torch.cat([x_fea_high_mid, x_fea_low], dim=1)
        x_fea_high_mid_low = self.high_mid_low_layer(x_fea_high_mid_low)
        out_prob = self.outp_layer(x_fea_high_mid_low)
        return out_prob

