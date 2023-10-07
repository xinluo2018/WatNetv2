## author: xin luo
## creat: 2022.4.3
## des: xception65 model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SeparableConv2d(nn.Module):
    ''' dilated_dw_conv2d -> bn(optional) -> 1x1 conv'''
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, \
                        stride, 0, dilation, groups=in_channels, bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

    def fix_padding(self, x, ksize, dilation):
        ksize_effective = ksize + (ksize - 1) * (dilation - 1)
        pad_total = ksize_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

class Block(nn.Module):
    ''' (relu(option) -> SeparableConv2d + bn)xn        
        grow (option for first or last): in_channels -> out_channels
        
        '''
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, norm_layer=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        filters = in_channels
        if grow_first:  #
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, \
                                        dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, \
                                        norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, \
                                        dilation, norm_layer=norm_layer))
        if stride != 1:  # specify a new SeparableConv2d in the last
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, stride, \
                                        norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        elif is_last:  # add a bn layer in the last
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, \
                                        dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out

class Xception65(nn.Module):
    '''Modified Aligned Xception
    output_stride = 32' means output image size is 1/32'''
    def __init__(self, num_bands=4, num_classes=1000, \
                            output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception65, self).__init__()
        ## output size:1/32
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        ## output size:1/16
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        ## output size:1/8
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        self.relu = nn.ReLU(True)
        # ---------Entry flow--------- #
        self.conv1 = nn.Conv2d(num_bands, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, \
                                start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, \
                                start_with_relu=False)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, \
                                norm_layer=norm_layer, start_with_relu=True, \
                                is_last=True)
        # ---------Middle flow--------- #
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,
                                 start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)
        # ---------Exit flow--------- #
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, \
                        dilation=exit_block_dilations[0],norm_layer=norm_layer,\
                        start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, \
                        dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, \
                        dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, \
                        dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        # c1 = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)
        # Middle flow
        x = self.midflow(x)
        # c3 = x
        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

