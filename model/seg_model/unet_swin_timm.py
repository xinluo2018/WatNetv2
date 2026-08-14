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
        self.DecoderBlocks = nn.ModuleList([
            conv3x3_bn_relu(self.out_channels[3], self.decode_channels[0]),   
            conv3x3_bn_relu(self.decode_channels[0] + self.out_channels[2], self.decode_channels[1]),  
            conv3x3_bn_relu(self.decode_channels[1] + self.out_channels[1], self.decode_channels[2]), 
            conv3x3_bn_relu(self.decode_channels[2] + self.out_channels[0], self.decode_channels[3])   
        ])
        
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
            conv3x3_bn_relu(self.decode_channels[3], 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
        ) 

    def forward(self, x):       
        '''
        x: input tensor, size: (B, num_bands, H, W)
        '''
        # Encoder 
        feas = self.encoder(x)  
        feas = [ feat.permute(0, 3, 1, 2) for feat in feas if feat is not None]
        # 提取最深层特征并融合 (Bottleneck)
        fea_fus = feas[-1]  
        fea_fus = self.DecoderBlocks[0](fea_fus)      
        fea_fus = self.up(fea_fus)  
        
        # Skip connections (反转前面的特征图列表，不包含最后一层)
        skips_fea = list(reversed(feas[:-1]))
        
        # 此时 skips_fea 长度为 3
        for i in range(len(skips_fea)):
            skip_fea = skips_fea[i]            
            fea_fus = torch.cat([fea_fus, skip_fea], dim=1)  
            fea_fus = self.DecoderBlocks[i+1](fea_fus)  
            if i < len(skips_fea) - 1:
                fea_fus = self.up(fea_fus)  
                
        # 最终上采样 (从 64x64 直接拉升回 256x256)
        logit = self.final_up(fea_fus)  
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