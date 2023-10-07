## author: xin luo
## create: 2021.6.29, modify: 2023.2.3
## des: the simple UNet model.


import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

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

def dwconv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, \
            kernel_size=3, stride=1, padding=1, groups=in_channels),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


####-----------for the unet-----------####
class dsample(nn.Module):
    '''down x2: pooling->conv_bn_relu->dwconv_bn_relu->conv_bn_relu
       down x4: pooling->conv_bn_relu->dwconv_bn_relu->dwconv_bn_relu->conv_bn_relu
    '''
    def __init__(self, in_channels, ex_channels, out_channels, scale = 2, **kwargs):
        super(dsample, self).__init__()
        self.scale = scale
        self.pool = nn.AvgPool2d(kernel_size=scale)
        self.conv_bn_relu_in = conv3x3_bn_relu(in_channels, ex_channels)
        self.dwconv_bn_relu_1 = dwconv3x3_bn_relu(ex_channels, ex_channels)
        self.dwconv_bn_relu_2 = dwconv3x3_bn_relu(ex_channels, ex_channels)
        self.conv_bn_relu_out = conv1x1_bn_relu(ex_channels, out_channels)
    def forward(self, x):
        if self.scale == 2:
            x = self.pool(x)
            x = self.conv_bn_relu_in(x)
            x = self.dwconv_bn_relu_1(x)
            x = self.conv_bn_relu_out(x)
        elif self.scale == 4:
            x = self.pool(x)
            x = self.conv_bn_relu_in(x) 
            x = self.dwconv_bn_relu_1(x)
            x = self.dwconv_bn_relu_2(x)
            x = self.conv_bn_relu_out(x)
        return x

class upsample(nn.Module):
    '''up x2: up_resize -> dwconv_bn_relu -> conv_bn_relu 
       up x4: up_resize -> dwconv_bn_relu -> dwconv_bn_relu -> conv_bn_relu 
    '''
    def __init__(self, in_channels, out_channels, scale = 2, **kwargs):
        super(upsample, self).__init__()
        self.scale = scale
        self.up2_layer = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4_layer = nn.Upsample(scale_factor=4, mode='nearest')
        self.dwconv_bn_relu_1 = dwconv3x3_bn_relu(in_channels, in_channels)
        self.dwconv_bn_relu_2 = dwconv3x3_bn_relu(in_channels, in_channels)
        self.conv_bn_relu_out = conv3x3_bn_relu(in_channels, out_channels)

    def forward(self, x):
        if self.scale == 2:
            x = self.up2_layer(x)
            x = self.dwconv_bn_relu_1(x)
            x = self.conv_bn_relu_out(x)
        elif self.scale == 4:
            x = self.up4_layer(x)
            x = self.dwconv_bn_relu_1(x)
            x = self.dwconv_bn_relu_2(x)
            x = self.conv_bn_relu_out(x)
        return x



class unet(nn.Module):
    ''' 
    description: unet model for single-scale image processing
    '''
    def __init__(self, num_bands, num_classes):
        super(unet, self).__init__()
        self.name = 'unet'
        self.num_classes = num_classes
        self.encoder = nn.ModuleList([
            dsample(in_channels=num_bands, ex_channels=32, out_channels=16, scale=2), # 1/2
            dsample(in_channels=16, ex_channels=64, out_channels=16, scale=2),   # 1/4
            dsample(in_channels=16, ex_channels=128, out_channels=32, scale=2),  # 1/8
            dsample(in_channels=32, ex_channels=128, out_channels=32, scale=4),  # 1/32
            dsample(in_channels=32, ex_channels=256, out_channels=64, scale=4),  # 1/128
        ])
        self.decoder = nn.ModuleList([
            upsample(in_channels=64, out_channels=64, scale=4),    # 1/32
            upsample(in_channels=64+32, out_channels=64, scale=4), # 1/8
            upsample(in_channels=64+32, out_channels=64, scale=2), # 1/4
            upsample(in_channels=64+16, out_channels=32, scale=2), # 1/2
        ])
        self.up_last = upsample(in_channels=32+16, out_channels=32, scale=2)
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                        nn.Sigmoid())
        else:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                        nn.Softmax(dim=1))

    def forward(self, input):
        x_encode = input
        '''feature encoding'''
        skips = []
        for encode in self.encoder:
            x_encode = encode(x_encode)
            skips.append(x_encode)
        skips = reversed(skips[:-1])

        '''feature decoding'''
        x_decode = x_encode
        for i, (decode, skip) in enumerate(zip(self.decoder, skips)):
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip], dim=1)
        output = self.up_last(x_decode)
        out_prob = self.outp_layer(output)
        return out_prob

