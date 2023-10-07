## author: xin luo
## creat: 2022.4.3
## des: mobilenet model

import torch.nn as nn
import numpy as np
from torchsummary import summary

def conv1x1_bn_relu6(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv3x3_bn_relu6(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    '''input size should be divided by 32 evenly'''
    def __init__(self, num_bands=4, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32   # first channel is always 32!
        last_channel = 1280
        self.last_channel = make_divisible(last_channel * width_mult) \
                                        if width_mult > 1.0 else last_channel
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],   
            [6, 24, 2, 2],      # size -> 1/4
            [6, 32, 3, 2],      # size -> 1/8
            [6, 64, 4, 2],      # size -> 1/16
            [6, 96, 3, 1],    
            [6, 160, 3, 2],     # size -> 1/32
            [6, 320, 1, 1],
        ]

        # ------head layer------ #
        self.head = conv3x3_bn_relu6(num_bands,input_channel,2)  # size -> 1/2

        # ------body layers (consist of inverted residual blocks)------ #
        # self.body = []
        self.body = nn.Sequential()
        for num, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            blocks = []  ## blocks in bottleneck
            for i in range(n):
                if i == 0:
                    blocks.append(block(input_channel, output_channel, s, expand_ratio=t))  
                else:
                    blocks.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
            self.body.add_module('inverted_'+str(num), nn.Sequential(*blocks))

        # ------tail layer------ # 
        # building last several layers
        self.tail = conv1x1_bn_relu6(input_channel, self.last_channel)

        # ------classifier------ #
        self.classifier = nn.Linear(self.last_channel, num_classes)
    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = x.mean(3).mean(2)   # global average pooling
        x = self.classifier(x)
        return x

