'''
author: xin luo
create: 2020.1.24
des: a simple U-Net model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# class GlobalBatchNorm2d(nn.Module):
#     """
#     One mean/var for the whole mini-batch over (N,C,H,W).
#     Optional affine: per-channel gamma/beta (or you can make them scalar).
#     """
#     def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
#         super().__init__()
#         self.eps = eps
#         self.affine = affine
#         if affine:
#             self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
#             self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
#         else:
#             self.register_parameter("weight", None)
#             self.register_parameter("bias", None)
#     def forward(self, x):
#         # x: (N,C,H,W)
#         mean = x.mean(dim=(0,1,2,3), keepdim=True)
#         var  = x.var(dim=(0,1,2,3), keepdim=True, unbiased=False)
#         x_hat = (x - mean) / torch.sqrt(var + self.eps)
#         if self.affine:
#             x_hat = x_hat * self.weight + self.bias
#         return x_hat


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        # GlobalBatchNorm2d(out_channels),
        # nn.GroupNorm(num_groups=1, num_channels=out_channels),
        nn.ReLU(inplace=True)
        )

class unet(nn.Module):
    def __init__(self, num_bands):
        super(unet, self).__init__()
        self.num_bands = num_bands
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down_conv1 = conv(self.num_bands, 16)
        self.down_conv2 = conv(16, 32)
        self.down_conv3 = conv(32, 64)
        self.down_conv4 = conv(64, 128)
        self.up_conv1 = conv(192, 64)
        self.up_conv2 = conv(96, 48)
        self.up_conv3 = conv(64, 32)
        self.outp = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                ) 

    def forward(self, x):   ## input size: 6x256x256
        ## encoder part
        x1 = self.down_conv1(x)              
        x1 = F.avg_pool2d(input=x1, kernel_size=2)  # 16x128x128
        x2 = self.down_conv2(x1)              
        x2 = F.avg_pool2d(input=x2, kernel_size=2) # 32x64x64
        x3 = self.down_conv3(x2)              
        x3 = F.avg_pool2d(input=x3, kernel_size=2) # 64x32x32
        x4 = self.down_conv4(x3)              
        x4 = F.avg_pool2d(input=x4, kernel_size=2) # 128x16x16
        ## decoder part
        x4_up = torch.cat([self.up(x4), x3], dim=1)  # (128+64)x32x32
        x3_up = self.up_conv1(x4_up)  # 64x32x32
        x3_up = torch.cat([self.up(x3_up), x2], dim=1)  # (64+32)x64x64
        x2_up = self.up_conv2(x3_up)  # 48x64x64
        x2_up = torch.cat([self.up(x2_up), x1], dim=1)  # (48+16)x128x128
        x1_up = self.up_conv3(x2_up)    # 32x128x128
        x1_up = self.up(x1_up)        # 32x256x256
        logit = self.outp(x1_up)
        return logit          

if __name__ == '__main__':
    model = unet(num_bands=7)
    input = torch.randn(1, 7, 256, 256)
    output = model(input)
    print(output.shape)