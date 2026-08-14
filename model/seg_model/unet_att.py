'''
author: xin luo
create: 2020.1.24
des: U-Net model with attention block (CBAM)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        # nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.ReLU(inplace=True)
        )

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel, 1, bias=True)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2, 1, 
                            kernel_size = kernel_size, 
                            padding = kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):
    def __init__(self, channel=256, reduction=4, kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel, reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual    ## residual connection

class unet_att(nn.Module):
    def __init__(self, num_bands, with_dem=True):
        super(unet_att, self).__init__()
        self.num_bands = num_bands
        self.with_dem = with_dem
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        if not self.with_dem:
            self.num_bands = num_bands-1
        self.att_2 = CBAMBlock(channel=32, reduction=4, kernel_size=7)
        self.att_3 = CBAMBlock(channel=64, reduction=4, kernel_size=7)
        self.att_4 = CBAMBlock(channel=128, reduction=4, kernel_size=7)
        self.down_conv1 = conv3x3_bn_relu(self.num_bands, 16)
        self.down_conv2 = conv3x3_bn_relu(16, 32)
        self.down_conv3 = conv3x3_bn_relu(32, 64)
        self.down_conv4 = conv3x3_bn_relu(64, 128)
        self.up_conv1 = conv3x3_bn_relu(192, 64)
        self.up_conv2 = conv3x3_bn_relu(96, 48)
        self.up_conv3 = conv3x3_bn_relu(64, 32)
        self.logit = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                ) 

    def forward(self, x):   ## input size: 6x256x256
        if not self.with_dem:
            x = x[:, :-1, :, :]
        ## encoder part
        x1 = self.down_conv1(x)              
        x1 = F.avg_pool2d(input=x1, kernel_size=2)  # 16x128x128
        x2 = self.down_conv2(x1)              # 32
        x2 = F.avg_pool2d(input=x2, kernel_size=2) # 32x64x64
        x2_att = self.att_2(x2)  # 
        x3 = self.down_conv3(x2_att)              # 64
        x3 = F.avg_pool2d(input=x3, kernel_size=2) # 64x32x32
        x3_att = self.att_3(x3)
        x4 = self.down_conv4(x3_att)              # 128
        x4 = F.avg_pool2d(input=x4, kernel_size=2) 
        x4_att = self.att_4(x4)
        ## decoder part
        x4_up = torch.cat([self.up(x4_att), x3_att], dim=1)  # (128+64)x32x32
        x3_up = self.up_conv1(x4_up)  # 64x32x32
        x3_up = torch.cat([self.up(x3_up), x2_att], dim=1)  # (64+32)x64x64
        x2_up = self.up_conv2(x3_up)  # 48x64x64
        x2_up = torch.cat([self.up(x2_up), x1], dim=1)  # (48+16)x128x128
        x1_up = self.up_conv3(x2_up)    # 32x128x128
        x1_up = self.up(x1_up)        # 32x256x256
        logit = self.logit(x1_up)
        return logit          

if __name__ == '__main__':
    model = unet_att(num_bands=7)
    input = torch.randn(1, 7, 256, 256)
    output = model(input)
    print(output.shape)