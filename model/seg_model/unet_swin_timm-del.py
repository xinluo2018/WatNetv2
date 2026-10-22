'''
author: xin luo
create: 2026.4.2
des: Dual-branch U-Net model with Swin Transformer V2 backbone
'''

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import torch.nn as nn
import timm
from model.mamba_seg.decoder import PyramidDecoder, UPerNetDecoder

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class unet_swin_timm(nn.Module):
    def __init__(self, num_bands=6, 
                       img_size=512,
                       backbone_name='swinv2_base_window8_256', 
                       pretrained=True):
        '''
        num_bands : 光学/多光谱分支波段数 (e.g., 6)
        '''
        super().__init__()
        self.num_bands = num_bands
        
        self.decode_channels = [512, 256, 128, 64] 
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        # ==========================================
        # 1. Encoder
        self.encoder = timm.create_model(backbone_name, 
                                        features_only=True, 
                                        img_size=img_size,
                                        in_chans=num_bands, 
                                        pretrained=pretrained)

        # 获取 encoder 每层的输出通道数
        self.out_channels = self.encoder.feature_info.channels()
        
        # ==========================================
        # 2. Decoder 部分 (针对 Swin 4个Stage 的通道拼装计算)
        # ==========================================
        self.decoder = PyramidDecoder(self.out_channels, 96, 1)

    def forward(self, x):       
        '''
        x: input tensor, size: (B, num_bands, H, W)
        '''
        # Encoder 
        feas = self.encoder(x)  
        feas = [ feat.permute(0, 3, 1, 2) for feat in feas if feat is not None]
        logit = self.decoder(feas)
        return logit

if __name__ == '__main__':
    # 模拟输入: Batch=2, 6个多光谱波段
    model = unet_swin_timm(num_bands=6, 
                         img_size = 512, 
                         backbone_name='swinv2_base_window8_256',
                         pretrained=True)
                         
    x = torch.randn(2, 6, 512, 512)  
    out = model(x)
    print("模型输出尺寸:", out.shape)  # 完美输出 [2, 1, 256, 256]