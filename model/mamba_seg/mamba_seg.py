'''
author: xin luo, 
created: 2026.10.2
des: integration of Mamba segmentation model
ref: https://github.com/mzeromiko/vmamba
'''

import math
import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model.mamba_seg.selective_scan import selective_scan_fn

## utils
class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)

class DropPath(nn.Module):
    """Per-sample stochastic depth without requiring timm."""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class Linear2d(nn.Linear):
    def forward(self, x):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        key = prefix + 'weight'
        if key in state_dict:
            state_dict[key] = state_dict[key].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, *args)

class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = Linear2d(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = Linear2d(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

## ss2d
def cross_scan_fn(x, force_torch=False):
    """Expand NCHW features into row, column and reversed sequences.
    x(N,C,H,W) -> y(N,4,C,H*W)
    """
    if x.is_cuda and not force_torch:
        from model.mamba_seg.selective_scan.triton.cross_scan import cross_scan_fn as triton_scan
        return triton_scan(x, force_torch=False)
    forward = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
    return torch.cat((forward, forward.flip(-1)), dim=1)


def cross_merge_fn(y, force_torch=False):
    """Align and sum four directional sequences."""
    if y.is_cuda and not force_torch:
        from model.mamba_seg.selective_scan.triton.cross_scan import cross_merge_fn as triton_merge
        return triton_merge(y, force_torch=False)
    batch, directions, channels, height, width = y.shape
    if directions != 4:
        raise ValueError('Expected four scan directions')
    y = y.view(batch, directions, channels, height * width)
    paired = y[:, :2] + y[:, 2:].flip(-1)
    rows = paired[:, 0]
    columns = paired[:, 1].view(batch, channels, width, height)
    return rows + columns.transpose(2, 3).flatten(2)


class SS2D(nn.Module):
    def __init__(self, d_model=96, d_state=1, ssm_ratio=1., dt_rank='auto',
                 dropout=0., backend='oflex'):
        super().__init__()
        self.k_group = 4
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else int(dt_rank)
        self.backend = backend

        self.in_proj = Linear2d(d_model, self.d_inner, bias=False)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, 3, padding=1,
            groups=self.d_inner, bias=False)
        self.act = nn.SiLU()
        projections = [nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False)
            for _ in range(self.k_group)]   ## linear projections for dt, B, and C, 
        self.x_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in projections]))  ## stacked weights for linear projections: (k_group, dt_rank + 2 * d_state, d_inner)
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = (
            mamba_init.init_dt_A_D(d_state, self.dt_rank, self.d_inner))
        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = Linear2d(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward_core(self, x):
        batch, channels, height, width = x.shape
        length = height * width
        xs = cross_scan_fn(x, force_torch=self.backend == 'torch')  # xs: (N, 4, C, H*W)
        ## Linear projection for dt, B, and C using grouped 1D convolution
        ## F.conv1d(): params->(input, weight, groups=1)
        ## output: [batch, 4*(dt_rank + 2 * d_state), length]
        x_dbl = F.conv1d(
                xs.view(batch, -1, length),                # -> (N, 4*C, H*W)
                self.x_proj_weight.view(-1, channels, 1),  # -> (4*(dt_rank + 2 * d_state), d_inner, 1)
                groups=self.k_group)
        # dts:(N, 4, dt_rank, length), Bs:(N, 4, d_state, length), Cs:(N, 4, d_state, length), 
        dts, Bs, Cs = x_dbl.view(batch, self.k_group, -1, length).split(
            (self.dt_rank, self.d_state, self.d_state), dim=2)
        dts = F.conv1d(
            dts.reshape(batch, -1, length),
            self.dt_projs_weight.view(self.k_group * channels, -1, 1), 
            groups=self.k_group)
        ## ys: (N, 4*C, H*W)
        ys = selective_scan_fn(
                    xs.view(batch, -1, length), 
                    dts,
                    -self.A_logs.float().exp(), 
                    Bs.contiguous(), 
                    Cs.contiguous(), 
                    self.Ds.float(), 
                    self.dt_projs_bias.flatten().float(),
                    backend=self.backend)
        merged = cross_merge_fn(
            ys.view(batch, self.k_group, channels, height, width),
            force_torch=self.backend == 'torch')
        return self.out_norm(merged.view(batch, channels, height, width)).to(x.dtype)

    def forward(self, x):
        x = self.act(self.conv2d(self.in_proj(x)))
        return self.dropout(self.out_proj(self.forward_core(x)))

### VSSBlock, i.e., vmamba block
class VSSBlock(nn.Module):
    """Official pre-norm SS2D and MLP residual structure."""
    def __init__(self, dim, 
                        drop_path=0.,
                        mlp_ratio=4., 
                        use_checkpoint=False,
                        ssm_d_state=1, 
                        ssm_ratio=1., 
                        ssm_dt_rank='auto',
                        backend='oflex', 
                        mixer_cls=None):
        super().__init__()
        if mixer_cls is None:
            mixer_cls = SS2D
        self.norm = LayerNorm2d(dim)
        self.op = mixer_cls(
                        dim, 
                        ssm_d_state, 
                        ssm_ratio, 
                        ssm_dt_rank, 
                        backend=backend)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        x = x + self.drop_path(self.op(self.norm(x)))  ## ln + ss2d + drop_path + residual
        return x + self.drop_path(self.mlp(self.norm2(x)))  ## ln + mlp + drop_path + residual

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

## mamba initialization
class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_min=0.001, dt_max=0.1):
        ## Initialize the linear projection for dt with the given rank and inner dimension
        projection = nn.Linear(dt_rank, d_inner, bias=True)
        bound = dt_rank ** -0.5
        nn.init.uniform_(projection.weight, -bound, bound)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        with torch.no_grad():
            projection.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        return projection

    @staticmethod
    def init_dt_A_D(d_state, dt_rank, d_inner, groups=4):
        ## Initialize the linear projections for dt, B, and C for each group
        projections = [mamba_init.dt_init(dt_rank, d_inner) for _ in range(groups)]
        dt_weight = nn.Parameter(torch.stack([layer.weight for layer in projections]))
        dt_bias = nn.Parameter(torch.stack([layer.bias for layer in projections]))

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_logs = nn.Parameter(A.log().view(1, 1, -1).repeat(groups, d_inner, 1)
                              .flatten(0, 1))
        Ds = nn.Parameter(torch.ones(groups * d_inner))
        A_logs._no_weight_decay = True
        Ds._no_weight_decay = True
        return A_logs, Ds, dt_weight, dt_bias

### mamba backbone: including patch embedding, vssblock and downsampling
def _patch_embed(in_chans, dim):
    """2 conv3, 4x downsample, 
    Official v2 patch embedding in NCHW format."""
    return nn.Sequential(
        nn.Conv2d(in_chans, dim // 2, 3, 2, 1), 
        nn.Identity(),
        LayerNorm2d(dim // 2), 
        nn.Identity(), 
        nn.GELU(),
        nn.Conv2d(dim // 2, dim, 3, 2, 1), 
        nn.Identity(), 
        LayerNorm2d(dim))

def _downsample(dim, out_dim):
    """ downsample to (1/2)*x, Official v3 downsampling in NCHW format."""
    return nn.Sequential(
        nn.Identity(), 
        nn.Conv2d(dim, out_dim, 3, 2, 1),
        nn.Identity(), 
        LayerNorm2d(out_dim))

class VMamba_Backbone(nn.Module):
    """Official ``v05_noz`` VMamba path with segmentation feature outputs."""
    def __init__(self, 
                    in_chans=3,
                    dims=96,   
                    depths=(2, 2, 8, 2),      ## mamba_tiny setting
                    out_indices=(0, 1, 2, 3), 
                    drop_path_rate=0.2,
                    ssm_d_state=1, 
                    ssm_ratio=1., 
                    ssm_dt_rank='auto', 
                    mlp_ratio=4.,
                    use_checkpoint=False, 
                    backend='oflex', 
                    pretrained=None,
                    block_cls=VSSBlock, 
                    block_kwargs=None, 
                    ssm_conv_bias=False,
                    forward_type='v05_noz', 
                    downsample_version='v3',
                    patchembed_version='v2', 
                    norm_layer='ln2d'):
        super().__init__()
        if len(depths) != 4 or tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError('Use four stages and ascending unique out_indices')
        if not out_indices or any(i not in range(4) for i in out_indices):
            raise ValueError('out_indices must select stages 0..3')
        official_path = (
            not ssm_conv_bias 
            and forward_type == 'v05_noz'
            and downsample_version == 'v3' 
            and patchembed_version == 'v2'
            and norm_layer == 'ln2d')
        if not official_path:
            raise ValueError('Only the official v05_noz/LN2D/v2/v3 path is supported')

        self.dims = [dims * 2**i for i in range(4)] if isinstance(dims, int) else list(dims)
        if len(self.dims) != 4:
            raise ValueError('dims must contain four widths')
        self.out_indices = tuple(out_indices)
        self.out_channels = tuple(self.dims[i] for i in self.out_indices)
        self.feature_strides = tuple(4 * 2**i for i in self.out_indices)
        self.pretrained = pretrained
        self.patch_embed = _patch_embed(in_chans, self.dims[0])

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.layers = nn.ModuleList()
        offset = 0
        for i, (dim, depth) in enumerate(zip(self.dims, depths)):
            blocks = [block_cls(
                        dim=dim, 
                        drop_path=rates[offset + j], 
                        mlp_ratio=mlp_ratio,
                        use_checkpoint=use_checkpoint, 
                        ssm_d_state=ssm_d_state,
                        ssm_ratio=ssm_ratio, 
                        ssm_dt_rank=ssm_dt_rank, 
                        backend=backend,
                        **(block_kwargs or {})) 
                    for j in range(depth)]
            ## downsample by 2x for all but the last stage
            downsample = _downsample(dim, self.dims[i + 1]) if i < 3 else nn.Identity() 
            self.layers.append(nn.Sequential(OrderedDict(
                    blocks=nn.Sequential(*blocks), downsample=downsample)))
            offset += depth  

        self.apply(self._init_weights)
        for i in self.out_indices:
            self.add_module(f'outnorm{i}', LayerNorm2d(self.dims[i]))
        self._pretrained_loaded = False

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def init_weights(self):
        if self.pretrained and not self._pretrained_loaded:
            self.load_pretrained(self.pretrained)
            self._pretrained_loaded = True

    def load_pretrained(self, path):
        """Load complete official classification or segmentation backbones."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state = {key.removeprefix('module.'): value for key, value in state.items()}

        if any(key.startswith('backbone.') for key in state):
            state = {key.removeprefix('backbone.'): value 
                     for key, value in state.items()
                     if key.startswith('backbone.')}
        state = {
                key: value for key, value in state.items()
                 if not key.startswith('classifier.')
                 }

        ## make the first convolution compatible with 6-channel input
        key = 'patch_embed.0.weight'
        if key in state:
                weight = state[key]
                target_channels = self.patch_embed[0].in_channels
                if weight.shape[1] == 3 and target_channels == 6:
                    state[key] = weight.repeat(1, 2, 1, 1) * 0.5

        expected = self.state_dict()
        missing = set(expected) - set(state)
        unexpected = set(state) - set(expected)
        allowed_missing = {key for key in expected if key.startswith('outnorm')}
        bad_shapes = {key: (tuple(value.shape), tuple(expected[key].shape))
                      for key, value in state.items()
                      if key in expected and value.shape != expected[key].shape}
        for key in list(bad_shapes):
            if (state[key].ndim == 4 and state[key].shape[-2:] == (1, 1)
                    and state[key].shape[:2] == expected[key].shape):
                state[key] = state[key].reshape(expected[key].shape)
                del bad_shapes[key]
        if missing - allowed_missing or unexpected or bad_shapes:
            raise RuntimeError(
                f'Checkpoint architecture mismatch: missing={sorted(missing-allowed_missing)}, '
                f'unexpected={sorted(unexpected)}, shapes={bad_shapes}')

        result = self.load_state_dict(state, strict=False)
        print(f'Loaded {path}; newly initialized output norms: {result.missing_keys}')
        return result

    def forward(self, x):
        x = self.patch_embed(x)
        outputs = []
        for i, stage in enumerate(self.layers):
            x = stage.blocks(x)
            if i in self.out_indices:
                outputs.append(getattr(self, f'outnorm{i}')(x).contiguous())
            x = stage.downsample(x)
        return outputs

### Segmentation    
class PyramidDecoder(nn.Module):
    def __init__(self, in_channels, channels, num_classes):
        super().__init__()
        if not in_channels or channels < 1 or num_classes < 1:
            raise ValueError('Require feature channels and positive decoder/classes widths')
        self.projections = nn.ModuleList(nn.Conv2d(c, channels, 1) for c in in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, num_classes, 1))

    def forward(self, features):
        if len(features) != len(self.projections):
            raise ValueError('Feature count does not match decoder projections')
        x = self.projections[-1](features[-1])  ## start from the last feature map, the last feature not to be downsampled
        for projection, feature in reversed(list(zip(self.projections[:-1], features[:-1]))):
            x = projection(feature) + F.interpolate(
                x, size=feature.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(x)


class MambaSeg(nn.Module):
    """
    des: Return full-resolution logits; either branch can be replaced.
        Custom backbones expose ``out_channels`` and return a sequence of NCHW maps.
        Custom decoders accept that sequence and return NCHW class logits.
    """
    def __init__(self, num_classes=2, 
                        in_chans=3, 
                        decoder_channels=128,
                        backbone=None, 
                        decoder=None, 
                        **backbone_kwargs):
        super().__init__()
        if num_classes < 1:
            raise ValueError('num_classes must be positive')
        if backbone is not None and backbone_kwargs:     # 
            raise ValueError('Configure a custom backbone before passing it in')
        self.backbone = backbone if backbone is not None else VMamba_Backbone(
            in_chans=in_chans, **backbone_kwargs)
        self.decoder = decoder if decoder is not None else PyramidDecoder(
            self.backbone.out_channels, decoder_channels, num_classes)
        if backbone is None:
            self.backbone.init_weights()

    def forward(self, x):
        logits = self.decoder(self.backbone(x))
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)


if __name__ == "__main__":
    # Test the model with dummy input
    model = MambaSeg(
        in_chans=6, 
        num_classes=1, 
        decoder_channels=32,
        dims=96, 
        depths=(2, 2, 8, 2), 
        ssm_d_state=1,
        backend='torch',
        # backend='oflex', 
        )
    # model = model.to('cuda')
    # dummy_input = torch.randn(4, 6, 512, 512, device='cuda')  # [B, C, H, W]
    dummy_input = torch.randn(4, 6, 512, 512)  # [B, C, H, W]
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Should be [1, num_classes]


