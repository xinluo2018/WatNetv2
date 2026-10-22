'''
author: xin luo
create: 2026.4.2
des: Dual-branch U-Net model with Swin Transformer V2 backbone
'''

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F


def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

### 1. PyramidDecoder
class PyramidDecoder(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        if not in_channels or channels < 1 or num_classes < 1:
            raise ValueError('Require feature channels and positive decoder/classes widths')
        self.projections = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in in_channels)
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            conv3x3_bn_relu(channels, 32),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
        ) 
    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError('Feature count does not match decoder projections')
        x = self.projections[-1](features[-1])
        for projection, feature in reversed(list(zip(self.projections[:-1], features[:-1]))):
            x = projection(feature) + F.interpolate(
                x, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)
    
class unet_backbone_timm(nn.Module):
    def __init__(self, num_bands=6, 
                    backbone_name='swinv2_base_window8_256', 
                    pretrained=True,
                    **kwargs
                    ):
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
                                        in_chans=num_bands, 
                                        pretrained=pretrained,
                                        **kwargs)

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
        ## make sure features are in NCHW format
        if any(feat.shape[1] != c for feat, c in zip(feas, self.out_channels)):
            feas = [feat.permute(0, 3, 1, 2).contiguous() for feat in feas]
        logit = self.decoder(feas)
        logit = F.interpolate(logit, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logit

if __name__ == '__main__':
    # 模拟输入: Batch=2, 6个多光谱波段
    model = unet_backbone_timm(num_bands=6, 
                            backbone_name='swinv2_base_window8_256',
                            # backbone_name = 'efficientnet_b0',
                            pretrained=True,
                            img_size = 512,  ## required for swin transformer model.
                            )
                         
    x = torch.randn(2, 6, 512, 512)  
    out = model(x)
    print("模型输出尺寸:", out.shape)  # 完美输出 [2, 1, 256, 256]





