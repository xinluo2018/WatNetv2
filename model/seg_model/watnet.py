## author: Dual-Star
## create: 2024.3.30;  
## modify by xin luo: 2024.5.20
## des: pytorch version watnet, 
## note: the watnet is very similar to the deeplabv3plus_mobilev2 in model_seg/deeplabv3plus_mobilev2.py; 
##       and the deeplabv3plus_mobilev2 have more parameters. 

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1_bn_relu6(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def conv3x3_bn_relu6(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True)
    )

def make_divisible(value, divisible_by=8):
    return int(math.ceil(value / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)
        hidden_channels = int(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1,
                          groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1,
                          groups=hidden_channels, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 implementation used by WatNet's feature extractor."""
    def __init__(self, num_bands=4, num_classes=1000, width_mult=1.0):
        super().__init__()
        input_channel = 32
        last_channel = 1280
        self.last_channel = (make_divisible(last_channel * width_mult)
                             if width_mult > 1.0 else last_channel)
        inverted_residual_setting = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        self.head = conv3x3_bn_relu6(num_bands, input_channel, 2)
        self.body = nn.Sequential()
        for index, (expand_ratio, channels, repeats, stride) in enumerate(
                inverted_residual_setting):
            output_channel = (make_divisible(channels * width_mult)
                              if expand_ratio > 1 else channels)
            blocks = []
            for repeat in range(repeats):
                block_stride = stride if repeat == 0 else 1
                blocks.append(InvertedResidual(
                    input_channel, output_channel, block_stride, expand_ratio))
                input_channel = output_channel
            self.body.add_module(f'inverted_{index}', nn.Sequential(*blocks))

        self.tail = conv1x1_bn_relu6(input_channel, self.last_channel)
        self.classifier = nn.Linear(self.last_channel, num_classes)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x = x.mean(3).mean(2)
        return self.classifier(x)


def aspp_conv(in_channels, out_channels, dilation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
                  dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class aspp_pooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.layers(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class aspp(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super().__init__()
        self.out_channels = 256
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU()
            ),
            aspp_conv(in_channels, self.out_channels, rate1),
            aspp_conv(in_channels, self.out_channels, rate2),
            aspp_conv(in_channels, self.out_channels, rate3),
            aspp_pooling(in_channels, self.out_channels),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(5 * self.out_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        return self.project(torch.cat([conv(x) for conv in self.convs], dim=1))

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

def deconv3x3_bn_relu(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, \
                            kernel_size=3, stride=2, padding=1, output_padding=1),    ### in tensorflow version watnet, kernel_size=3
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class mobilenet_feat(nn.Module):
    ## Get the 3rd, 9th, 39th-layers feature. 
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = MobileNetV2(num_bands=self.in_channels, num_classes=2)

    def forward(self, input):
        x = self.backbone.head(input)
        # Extract low-dimensional features
        x = self.backbone.body.inverted_0(x)      # channel -> 16, size -> 1/2,
        low_feat = x
        # Extract mid-dimensional features
        x = self.backbone.body.inverted_1(x)      # channel -> 24, size -> 1/4
        mid_feat = x
        # Extract high-dimensional features
        x = self.backbone.body.inverted_2(x)
        x = self.backbone.body.inverted_3(x)
        x = self.backbone.body.inverted_4(x)      #   channel -> 96, size -> 1/16
        high_feat = x
        return low_feat, mid_feat, high_feat

class watnet(nn.Module):
    def __init__(self, num_bands, 
                num_classes=2,
                aspp_atrous_rates=(6, 12, 18)):
        super().__init__()
        self.name = 'watnet'
        self.in_channels = num_bands
        self.channels_feas_mobilenet = [16, 24, 96]   ## the channels of low, mid, and high-level features.
        self.atrous_rates = aspp_atrous_rates
        # get multiscale features. 
        self.backbone = mobilenet_feat(self.in_channels)
        self.aspp = aspp(in_channels=self.channels_feas_mobilenet[2], atrous_rates=self.atrous_rates)
    
        self.mid_layer = conv1x1_bn_relu(self.channels_feas_mobilenet[1], 48)
        self.high_mid_layer = nn.Sequential(
                        conv3x3_bn_relu(48+self.aspp.out_channels, 128),
                        conv3x3_bn_relu(128, 128)
                        )
        self.low_layer = conv1x1_bn_relu(self.channels_feas_mobilenet[0], 48)
        self.high_mid_low_layer = nn.Sequential(
                        deconv3x3_bn_relu(128+48, 256),
                        nn.Dropout(0.5),
                        conv1x1_bn_relu(256, 128),
                        conv3x3_bn_relu(128, 128),
                        nn.Dropout(0.1),
                        )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1)
                    )
        else: 
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
                    )
        # Initialize model parameters.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')

    def forward(self, input):
        fea_low, fea_mid, fea_high = self.backbone(input)
        ### ------high level feature
        x_high = self.aspp(fea_high)            # output channels:256
        x_high = F.interpolate(x_high, fea_mid.size()[-2:], mode='bilinear', align_corners=True)
        ### ------ mid-level feature, and concatenate high level feature.
        x_mid = self.mid_layer(fea_mid)
        x_high_mid = torch.cat([x_high, x_mid], dim=1)
        x_high_mid = self.high_mid_layer(x_high_mid)
        x_high_mid = F.interpolate(x_high_mid, fea_low.size()[-2:], mode='bilinear', align_corners=True)
        ### ------low-level feature, and concatenate high and mid level features.
        x_low = self.low_layer(fea_low)
        x_high_mid_low = torch.cat([x_high_mid, x_low], dim=1)
        x_high_mid_low = self.high_mid_low_layer(x_high_mid_low)
        ### output layer
        out_logit = self.outp_layer(x_high_mid_low)
        return out_logit

if __name__ == '__main__':
    model = watnet(num_bands=6, num_classes=2)
    print(model)
    input_tensor = torch.randn(1, 6, 512, 512)   # Example input tensor with shape (batch_size, num_bands, height, width)
    output = model(input_tensor)
    print(output.shape)  # Output shape


