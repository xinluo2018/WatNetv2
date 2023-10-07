## author: xin luo
## creat: 2022.4.3, modify: 2023.2.3
## des: deeplabv3plus model (with Xception65 backbone).


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone.xception65 import Xception65


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

class Xception65_feat(nn.Module):
    '''original encoder for deeplabv3_plus moel
      retrun:
        high-level + low-level features

    '''
    def __init__(self, num_bands=4):
        super(Xception65_feat, self).__init__()
        self.backbone = Xception65(num_bands)
    def forward(self,x):
        '''output -> low feature + high feature'''
        ## output_stride = 32
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.relu(x)
        x = self.backbone.block1(x)
        x = self.backbone.relu(x)    
        fea_low = x      # size -> 1/4, num_channel -> 128
        x = self.backbone.block2(x)
        x = self.backbone.block3(x)
        # Middle flow
        x = self.backbone.midflow(x)
        # fea_mid = x      # size -> 1/16, num_channel -> 728
        # Exit flow
        x = self.backbone.block20(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv3(x)
        x = self.backbone.bn3(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv4(x)
        x = self.backbone.bn4(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv5(x)
        x = self.backbone.bn5(x)
        x = self.backbone.relu(x) 
        fea_high = x      # size -> 1/32, num_channels -> 2048
        return fea_low, fea_high



class deeplabv3plus(nn.Module):
    '''des: original deeplabv3_plus model'''
    def __init__(self, num_bands=4, num_classes=2, backbone=Xception65_feat):
        super(deeplabv3plus, self).__init__()
        self.name = 'deeplabv3plus'
        self.backbone = backbone(num_bands=num_bands)
        self.aspp = aspp(in_channels=2048, atrous_rates=[12, 24, 36])
        self.low_block = conv1x1_bn_relu(128, 48)
        self.high_low_block = nn.Sequential(      
                            conv3x3_bn_relu(304, 256),
                            nn.Dropout(0.5),
                            conv3x3_bn_relu(256, 256),
                            nn.Dropout(0.1),
                            )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                    nn.Sigmoid())
        else:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=1),
                    nn.Softmax(dim=1))


    def forward(self, x):
        fea_low, fea_high = self.backbone(x) 
        ## -------high-level features------- ##
        x_fea_high = self.aspp(fea_high)    # channels: -> 256
        x_fea_high = F.interpolate(x_fea_high, fea_low.size()[-2:], \
                                    mode='bilinear', align_corners=True)

        ## -------low-level features------- ##
        x_fea_low = self.low_block(fea_low)   # channels: ->48

        ## -------features concat-------  ##
        x_fea_concat = torch.cat([x_fea_high, x_fea_low], dim=1) # 
        x_fea_concat = self.high_low_block(x_fea_concat)
        x_out_prob = self.outp_layer(x_fea_concat)
        x_out_prob = F.interpolate(x_out_prob, x.size()[-2:], \
                                        mode='bilinear', align_corners=True)
        return x_out_prob

