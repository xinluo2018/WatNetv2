'''
author: xin luo
create: 2026.4.24
des: UNet-like swin transformer model for segmentation task
ref:(1) https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
    (2) https://github.com/amarbit/SwinTransformerFromScratch
# UNDO: use einops to simplify tensor reshaping and permutation 
'''

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.layers.weight_init import trunc_normal_

class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
    
class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings"""    
    def __init__(self, img_size: int = 224, 
                            patch_size: int = 4, 
                            in_channels: int = 3, 
                            embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # Project patches to embedding dimension
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape  
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model's expected size ({self.img_size}*{self.img_size})."
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        x = self.norm(x)
        return x

class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA) or Shifted-Window MSA (SW-MSA)"""
    
    def __init__(self, dim: int, 
                        window_size: int, 
                        num_heads: int, 
                        shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.shift_size = shift_size
        assert dim % num_heads == 0,\
              "Embedding dimension must be divisible by number of heads."
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))
        relative_coords_table = self._relative_coords_table(window_size)  # [window_size, window_size, 2]
        self.register_buffer("relative_coords_table", relative_coords_table)
        relative_position_index = self._relative_position_index(window_size)
        self.register_buffer("relative_position_index", relative_position_index)

    def _relative_coords_table(self, window_size: int) -> torch.Tensor:
        """Generate relative coordinate table for a given window size"""
        relative_coords_h = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size - 1), self.window_size, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, 
                            relative_coords_w], indexing='ij')
                            ).permute(1, 2, 0).contiguous().unsqueeze(0)  # [1, 2*window_size-1, 2*window_size-1, 2]
        relative_coords_table[:,:,:,0] /= self.window_size - 1  # Normalize to [-1, 1]
        relative_coords_table[:,:,:,1] /= self.window_size - 1
        relative_coords_table *= 8  # Scale to increase resolution of relative position bias
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)  # Logarithmic scaling
        return relative_coords_table  # [1, 2*window_size-1, 2*window_size-1, 2]

    def _relative_position_index(self, window_size: int) -> torch.Tensor:
        """Generate relative position index for each token inside the window"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # [2, window_size, window_size]
        coords_flatten = torch.flatten(coords, 1)  # [2, window_size*window_size]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, window_size*window_size, window_size*window_size]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [window_size*window_size, window_size*window_size, 2]
        relative_coords[:, :, 0] += self.window_size - 1  # Shift to ensure non-negative indices
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1  # Combine x and y relative positions into a single index: index = row * width + col
        relative_position_index = relative_coords.sum(-1)  # [window_size*window_size, window_size*window_size]
        return relative_position_index

    def forward(self, x: torch.Tensor, 
                    mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]        
        # Add relative position bias
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads) # (2*Wh-1)*(2*Ww-1), nH
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask if provided (for shifted window)
        if mask is not None:
            nW = mask.shape[0] # Number of windows, [total_windows, window_size*window_size, window_size*window_size]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with W-MSA and SW-MSA"""
    def __init__(self, dim: int, 
                        num_heads: int, 
                        input_resolution: tuple[int, int],
                        window_size: int, 
                        shift_size: int = 0, 
                        mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Window attention
        self.window_attn = WindowAttention(dim, window_size, num_heads, shift_size)
        
        # MLP
        self.mlp = MLP(dim, mlp_ratio)
        # Attention mask for SW-MSA
        if shift_size > 0:
            attn_mask = self._create_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def _create_mask(self, input_resolution: tuple[int, int]) -> torch.Tensor:
        """Create mask for shifted window attention
        """
        H, W = input_resolution
        # Ensure dimensions are compatible
        shift_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:    
            for w in w_slices:    
                shift_mask[:, h, w, :] = cnt   # Assign a unique mask value to each region
                cnt += 1  
        shift_mask_windows = self._window_partition(shift_mask, self.window_size)  # [num_windows, window_size, window_size, 1]
        shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size) # [total_windows, window_size*window_size]
        attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # [total_windows, window_size*window_size, window_size*window_size]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def _window_partition(self, x: torch.Tensor, window_size: int) -> torch.Tensor:
        """ Partition into non-overlapping windows """
        B, H, W, C = x.shape        
        # If feature map is smaller than window size, pad it
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)        
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C) 
        return windows  ## [num_windows*B, window_size, window_size, C]
    
    def _window_reverse(self, windows: torch.Tensor, 
                       window_size: int, H: int, W: int) -> torch.Tensor:
        """ Reverse window partition """
        # Calculate number of windows
        num_windows_h = H // window_size   # upper bound for number of windows in height
        num_windows_w = W // window_size   
        B = windows.shape[0] // (num_windows_h * num_windows_w)        
        x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)        
        return x
    
    def _att(self, x: torch.Tensor) -> torch.Tensor:
        """Apply window attention to input feature map"""
        # B, H, W, C = x.shape
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, H, W, C)  # Reshape to [B, H, W, C]
        # Cyclic shift for shifted window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x        
        # Window partition
        x_windows = self._window_partition(shifted_x, self.window_size) # [total_windows*B, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        # Window attention: W-MSA / SW-MSA
        attn_windows = self.window_attn(x_windows, self.attn_mask)        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = self._window_reverse(attn_windows, self.window_size, H, W)        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x        
        x = x.view(B, H * W, C)  # Reshape back to [B, H*W, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: [B, H*W, C]'''
        shortcut = x  # [B, H*W, C]
        x = self.norm1(x)        
        # Attention
        x = self._att(x)  # [B, H*W, C]
        x = shortcut + x    # Residual connection after attention
        # FFN
        x = x + self.mlp(self.norm2(x))    # [B, H*W, C]     
        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, 
                 dim, 
                 input_resolution: tuple[int, int], 
                 depth, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.window_size = window_size
        assert 0 <= self.window_size <= min(self.input_resolution), \
            f"Window size {self.window_size} is larger than input resolution {self.input_resolution}"
        assert all(i % window_size == 0 for i in self.input_resolution), \
            f"Input resolution {self.input_resolution} must be divisible by window size {window_size}"
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, 
                                 num_heads=num_heads,
                                 input_resolution=self.input_resolution,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 )
                for i in range(depth)])

    def forward(self, x):
        B, L, C = x.shape
        for blk in self.blocks:
            x = blk(x)
        return x

class PatchMerging(nn.Module):
    """Patch Merging Layer"""
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: B, H*W, C -> B, H/2*W/2, 2*C'''
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)        
        assert H % 2 == 0 and W % 2 == 0, \
            f"Input feature map size ({H}x{W}) must be even for patch merging."
        # Merge patches in 2x2 neighborhoods
        x = x.view(B, H // 2, 2, W // 2, 2, C)  # [B, H/2, 2, W/2, 2, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous() # [B, H/2, W/2, 2, 2, C]
        x = x.view(B, -1, 4 * C)  
        x = self.norm(x)
        x = self.reduction(x)   # [B, H/2*W/2, 2*C]
        return x

class PatchExpand(nn.Module):
    """ Patch Expand Layer.
    Args:
        input_resolution (tuple[int]): height and width of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim//2)
    def forward(self, x):
        """ x: [B, H*W, C] -> [B, 4*H*W, C//2] """ 
        H, W = self.input_resolution
        B, L, C = x.shape
        x = self.expand(x)  # B H*W 2*C
        x = x.view(B, H, W, 2, 2, C//2)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, 4*H*W , C//2)
        x = self.norm(x)    # [B, 4*H*W, C//2]
        return x

class FinalPatchExpansion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, 16*dim, bias=False)
    def forward(self, x):
        ''' x: [B, H*W, dim] -> [B, 16*H*W, dim] '''
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H , W, 4, 4, C//16)
        x = x.permute(0,1,3,2,4,5)
        x = x.reshape(B, 16*H*W , C//16)
        x = self.norm(x) 
        return x

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, 
                 patch_size=4, 
                 num_class=10,
                 in_chans=1, 
                 embed_dim=96, 
                 window_size=7):
        super().__init__()
        self.image_size = img_size
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                                in_channels=in_chans, embed_dim=embed_dim)
        input_resolution = (img_size // patch_size, img_size // patch_size)
        H, W = input_resolution
        ## build encoder layers
        self.encoder_layers = nn.ModuleList([
            BasicLayer(dim=1*embed_dim, input_resolution=(H, W), depth=2, num_heads=3,  window_size=window_size, mlp_ratio=4),
            BasicLayer(dim=2*embed_dim, input_resolution=(H//2, W//2), depth=2, num_heads=6, window_size=window_size, mlp_ratio=4),
            BasicLayer(dim=4*embed_dim, input_resolution=(H//4, W//4), depth=2, num_heads=12, window_size=window_size, mlp_ratio=4)
            ])
        self.encoder_merge_layers = nn.ModuleList([
            PatchMerging(1*embed_dim),      # out: [B, H/2*W/2, 2*embed_dim]
            PatchMerging(2*embed_dim),      # out: [B, H/4*W/4, 4*embed_dim]
            PatchMerging(4*embed_dim)       # out: [B, H/8*W/8, 8*embed_dim]
            ])
        self.bottleneck = BasicLayer(dim=8*embed_dim, input_resolution=(H//8, W//8), 
                                        depth=2, num_heads=12, window_size=window_size, mlp_ratio=4)
        ## build decoder layers
        self.decoder_expand_layers = nn.ModuleList([
            PatchExpand((H//8, W//8), 8*embed_dim),  # out: [B, H/4*W/4, 4*embed_dim]
            PatchExpand((H//4, W//4), 4*embed_dim),  # out: [B, H/2*W/2, 2*embed_dim]
            PatchExpand((H//2, W//2), 2*embed_dim)   # out: [B, H*W, embed_dim]
        ])  
        self.skip_conn_concat = nn.ModuleList([
            nn.Linear(8*embed_dim, 4*embed_dim),   # out: [B, H/4*W/4, 4*embed_dim]
            nn.Linear(4*embed_dim, 2*embed_dim),   # out: [B, H/2*W/2, 2*embed_dim]
            nn.Linear(2*embed_dim, 1*embed_dim)    # out: [B, H*W, embed_dim]
        ])
        self.decoder_layers = nn.ModuleList([
            BasicLayer(dim=4*embed_dim, input_resolution=(H//4, W//4), depth=2, num_heads=12, window_size=window_size, mlp_ratio=4),
            BasicLayer(dim=2*embed_dim, input_resolution=(H//2, W//2), depth=2, num_heads=6, window_size=window_size, mlp_ratio=4),
            BasicLayer(dim=1*embed_dim, input_resolution=(H, W), depth=2, num_heads=3, window_size=window_size, mlp_ratio=4)
        ])

        self.final_patch_expansion = FinalPatchExpansion(embed_dim) # out: [B, 16*H*W, embed_dim]
        if num_class > 2:
            self.head = nn.Conv2d(in_channels=embed_dim, out_channels=num_class, kernel_size=1)
        else:
            self.head = nn.Conv2d(in_channels=embed_dim, out_channels=1, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):        
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        skip_feas = []
        ## encoder
        for encode_layer, encode_merge in zip(self.encoder_layers, self.encoder_merge_layers):
            x = encode_layer(x)
            skip_feas.append(x)
            x = encode_merge(x)
        x = self.bottleneck(x)
        ## decoder
        skip_feas = skip_feas[::-1]  # reverse the order of skip features for decoding        
        for i, decode_layer in enumerate(self.decoder_layers):
            skip_fea = skip_feas[i]
            x = self.decoder_expand_layers[i](x)
            x = torch.cat([x, skip_fea], dim=-1)
            x = self.skip_conn_concat[i](x)
            x = decode_layer(x)
        x = self.final_patch_expansion(x)
        B, L, C = x.shape
        x = x.reshape(B, self.image_size, self.image_size, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = self.head(x)
        return x

if __name__ == "__main__":
    # Test the model with dummy input
    model = SwinUNet(img_size=224, 
                     patch_size=4, 
                     num_class=2, 
                     in_chans=6, 
                     embed_dim=96, 
                     window_size=7)
    dummy_input = torch.randn(4, 6, 224, 224)  # [B, C, H, W]
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be [1, num_classes]

