'''
author: xin luo
create: 2026.4.2
des: U-Net model with timm backbone
'''

import torch
import torch.nn as nn
import timm

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class unet_timm(nn.Module):
    def __init__(self, num_bands, 
                        backbone_name='resnet34', 
                        pretrained=False):
        '''
        num_bands: number of input bands
        num_bands_b2: number of bands for branch 2 (e.g., DEM)
        '''
        super().__init__()
        self.num_bands = num_bands
        self.decode_channels = [64, 64, 64, 64, 32]  # decoder channels for each stage
        self.up = nn.Upsample(scale_factor=2, mode='nearest')  # upsample layer

        ## encoder part
        self.encoder = timm.create_model(backbone_name, 
                                        features_only=True, 
                                        in_chans=num_bands, 
                                        pretrained=pretrained)

        self.out_channels = self.encoder.feature_info.channels()
        ## decoder part (fused features)
        self.DecoderBlocks = nn.ModuleList([
                conv3x3_bn_relu(self.out_channels[-1], self.decode_channels[0]),   
                conv3x3_bn_relu(self.out_channels[-2]+self.decode_channels[0], self.decode_channels[1]),  
                conv3x3_bn_relu(self.out_channels[-3]+self.decode_channels[1], self.decode_channels[2]), 
                conv3x3_bn_relu(self.out_channels[-4]+self.decode_channels[2], self.decode_channels[3]), 
                conv3x3_bn_relu(self.out_channels[-5]+self.decode_channels[3], self.decode_channels[4])   
                ])
        self.logit = nn.Sequential(
                        nn.Conv2d(self.decode_channels[4], 1, kernel_size=3, padding=1),
                        )
    def forward(self, x):       ## input size: 6x256x256
        '''
        x: input tensor
        '''
        ## encoder part
        feas = self.encoder(x)  # list of features from encoder
        fea = feas[-1]   #   
        fea_fus = self.DecoderBlocks[0](fea)   # fused features through decoder    
        fea_fus = self.up(fea_fus)  # upsample to match next skip connection
        # skip connections: 
        skips_fea = list(reversed(feas[:-1]))
        for i, skip_fea in enumerate(skips_fea):
            fea_fus = torch.cat([fea_fus, skip_fea], dim=1)  # concat skip features
            fea_fus = self.DecoderBlocks[i+1](fea_fus)  # decode fused features
            fea_fus = self.up(fea_fus)  # upsample for next stage
        logit = self.logit(fea_fus)  # 1x256x256
        return logit

if __name__ == '__main__':
    model = unet_timm(num_bands=6, 
                        # backbone_name='resnet34', 
                        backbone_name='efficientnet_b0',
                        pretrained=True)
    x = torch.randn(2, 6, 256, 256)  # batch_size=2, num_bands=6, H=W=256
    out = model(x)
    print(out.shape)  # should be [2, 1, 256, 256] 


