## author: xin luo
## create: 2021.6.29, modify: 2023.2.3
## des: multiscale UNet model (with multiscale input).


import torch
import torch.nn as nn
import torch.nn.functional as F

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

def convert_g_l(img_g, scale_ratio):
    '''global_size should be divisible by local_size.
    '''
    size_g = img_g.shape[2]
    size_l = size_g/scale_ratio
    if size_l >= 1:
        ''' crop -> enlarge scale '''
        start_crop = int((size_g - size_l)//2)
        img_l_crop = img_g[:,:, start_crop:int(start_crop+size_l), start_crop:int(start_crop+size_l)]
        img_l = F.interpolate(img_l_crop, size=[size_g, size_g], mode='nearest')

    else:
        ''' enlarge scale -> crop '''
        start_crop = int((size_g*scale_ratio - size_l*scale_ratio)//2)
        img_g_up = F.interpolate(img_g, size=[size_g*scale_ratio, size_g*scale_ratio], mode='nearest')
        img_l = img_g_up[:, :, start_crop:start_crop+int(size_l*scale_ratio), \
                                    start_crop:start_crop+int(size_l*scale_ratio)]

    return img_l

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

class unet_scales(nn.Module):
    ''' 
    description: unet model for single-scale image processing
    '''
    def __init__(self, num_bands, num_classes, scale_high=2048, scale_mid=512, scale_low=256):
        super(unet_scales, self).__init__()
        self.name = 'unet_scales'
        self.num_classes = num_classes
        self.scale_high, self.scale_mid, self.scale_low = scale_high, scale_mid, scale_low
        self.high2low_ratio = scale_high//scale_low
        self.mid2low_ratio = scale_mid//scale_low
        self.encoder = nn.ModuleList([
            dsample(in_channels=num_bands, ex_channels=32, out_channels=16, scale=2), # 1/2
            dsample(in_channels=16, ex_channels=64, out_channels=16, scale=2),   # 1/4
            dsample(in_channels=16, ex_channels=128, out_channels=32, scale=2),  # 1/8
            dsample(in_channels=32, ex_channels=128, out_channels=32, scale=4),  # 1/32
            dsample(in_channels=32, ex_channels=256, out_channels=64, scale=4),  # 1/128
        ])
        self.decoder = nn.ModuleList([
            upsample(in_channels=64*3, out_channels=64, scale=4),         # 1/32
            upsample(in_channels=64+32*3, out_channels=64, scale=4),      # 1/8
            upsample(in_channels=64+32*3, out_channels=64, scale=2),      # 1/4
            upsample(in_channels=64+16*3, out_channels=32, scale=2),      # 1/2
        ])
        self.up_last = upsample(in_channels=32+16*3, out_channels=32, scale=2)
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1),
                        nn.Sigmoid())
        else:
            self.outp_layer = nn.Sequential(
                        nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1),
                        nn.Softmax(dim=1))

    def forward(self, input):
        x_encode_high, x_encode_mid, x_encode_low = input[0], input[1], input[2]

        '''--------feature encoding---------'''
        skips_high, skips_mid, skips_low= [],[],[]

        '''---low-level feature learning'''
        for encode in self.encoder:
            x_encode_low = encode(x_encode_low)
            skips_low.append(x_encode_low)
        skips_low = reversed(skips_low[:-1])

        '''---mid-level feature learning'''
        for encode in self.encoder:
            x_encode_mid = encode(x_encode_mid)
            x_encode_mid2low = convert_g_l(img_g=x_encode_mid, \
                                            scale_ratio=self.mid2low_ratio)
            skips_mid.append(x_encode_mid2low)
        skips_mid = reversed(skips_mid[:-1])

        '''---high-level feature learning'''
        for encode in self.encoder:
            x_encode_high = encode(x_encode_high)
            x_encode_high2low = convert_g_l(img_g=x_encode_high, \
                                            scale_ratio=self.mid2low_ratio)
            skips_high.append(x_encode_high2low)
        skips_high = reversed(skips_high[:-1])

        '''-------- feature decoding-------'''
        '''--- feature fusion'''
        x_decode = torch.cat((x_encode_low, x_encode_mid2low, x_encode_high2low), 1)
        # x_decode = x_encode_low

        for i, (decode, skip_high, skip_mid, skip_low) in enumerate(zip(self.decoder, skips_high, skips_mid, skips_low)):
            x_decode = decode(x_decode)
            x_decode = torch.cat([x_decode, skip_high, skip_mid, skip_low], dim=1)
            # x_decode = torch.cat([x_decode, skip_low], dim=1)

        output = self.up_last(x_decode)
        out_prob = self.outp_layer(output)
        return out_prob

