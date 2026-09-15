
# Adapted from MzeroMiko/VMamba (MIT); see LICENSE.
"""Four-stage VMamba segmentation backbone."""
from collections import OrderedDict

import torch
from torch import nn

from .vss_block import LayerNorm2d, VSSBlock


def _patch_embed(in_chans, dim):
    """Official v2 patch embedding in NCHW format."""
    return nn.Sequential(
        nn.Conv2d(in_chans, dim // 2, 3, 2, 1), nn.Identity(),
        LayerNorm2d(dim // 2), nn.Identity(), nn.GELU(),
        nn.Conv2d(dim // 2, dim, 3, 2, 1), nn.Identity(), LayerNorm2d(dim))


def _downsample(dim, out_dim):
    """Official v3 downsampling in NCHW format."""
    return nn.Sequential(
        nn.Identity(), nn.Conv2d(dim, out_dim, 3, 2, 1),
        nn.Identity(), LayerNorm2d(out_dim))


class Backbone_VSSM(nn.Module):
    """Official ``v05_noz`` VMamba path with segmentation feature outputs."""

    def __init__(self, dims=96, depths=(2, 2, 8, 2), in_chans=3,
                 out_indices=(0, 1, 2, 3), drop_path_rate=0.2,
                 ssm_d_state=1, ssm_ratio=1., ssm_dt_rank='auto', mlp_ratio=4.,
                 use_checkpoint=False, backend='oflex', pretrained=None,
                 block_cls=VSSBlock, block_kwargs=None, ssm_conv_bias=False,
                 forward_type='v05_noz', downsample_version='v3',
                 patchembed_version='v2', norm_layer='ln2d'):
        super().__init__()
        if len(depths) != 4 or tuple(sorted(set(out_indices))) != tuple(out_indices):
            raise ValueError('Use four stages and ascending unique out_indices')
        if not out_indices or any(i not in range(4) for i in out_indices):
            raise ValueError('out_indices must select stages 0..3')
        official_path = (
            not ssm_conv_bias and forward_type == 'v05_noz'
            and downsample_version == 'v3' and patchembed_version == 'v2'
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
                dim=dim, drop_path=rates[offset + j], mlp_ratio=mlp_ratio,
                use_checkpoint=use_checkpoint, ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio, ssm_dt_rank=ssm_dt_rank, backend=backend,
                **(block_kwargs or {})) for j in range(depth)]
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
            state = {key.removeprefix('backbone.'): value for key, value in state.items()
                     if key.startswith('backbone.')}
        state = {key: value for key, value in state.items()
                 if not key.startswith('classifier.')}

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


class VMamba(Backbone_VSSM):
    """Short project-facing name for ``Backbone_VSSM``."""
