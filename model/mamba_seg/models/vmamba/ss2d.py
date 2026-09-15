# Adapted from MzeroMiko/VMamba (MIT); see LICENSE.
"""Official VMamba ``v05_noz`` SS2D path."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from ...ops.selective_scan import selective_scan_fn
from .vss_block import LayerNorm2d, Linear2d


def cross_scan_fn(x, force_torch=False):
    """Expand NCHW features into row, column and reversed sequences."""
    if x.is_cuda and not force_torch:
        from ...ops.selective_scan.triton.cross_scan import cross_scan_fn as triton_scan
        return triton_scan(x, force_torch=False)
    forward = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
    return torch.cat((forward, forward.flip(-1)), dim=1)


def cross_merge_fn(y, force_torch=False):
    """Align and sum four directional sequences."""
    if y.is_cuda and not force_torch:
        from ...ops.selective_scan.triton.cross_scan import cross_merge_fn as triton_merge
        return triton_merge(y, force_torch=False)
    batch, directions, channels, height, width = y.shape
    if directions != 4:
        raise ValueError('Expected four scan directions')
    y = y.view(batch, directions, channels, height * width)
    paired = y[:, :2] + y[:, 2:].flip(-1)
    rows = paired[:, 0]
    columns = paired[:, 1].view(batch, channels, width, height)
    return rows + columns.transpose(2, 3).flatten(2)


# Small wrappers retained for direct tests and callers.
def cross_scan(x):
    return cross_scan_fn(x, force_torch=True)


def cross_merge(y, height, width):
    merged = cross_merge_fn(
        y.view(*y.shape[:3], height, width), force_torch=True)
    return merged.view(y.shape[0], y.shape[2], height, width)


class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_min=0.001, dt_max=0.1):
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
            for _ in range(self.k_group)]
        self.x_proj_weight = nn.Parameter(
            torch.stack([layer.weight for layer in projections]))
        self.A_logs, self.Ds, self.dt_projs_weight, self.dt_projs_bias = (
            mamba_init.init_dt_A_D(d_state, self.dt_rank, self.d_inner))
        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = Linear2d(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward_core(self, x):
        batch, channels, height, width = x.shape
        length = height * width
        xs = cross_scan_fn(x, force_torch=self.backend == 'torch')
        projected = F.conv1d(
            xs.view(batch, -1, length),
            self.x_proj_weight.view(-1, channels, 1), groups=self.k_group)
        dts, Bs, Cs = projected.view(batch, self.k_group, -1, length).split(
            (self.dt_rank, self.d_state, self.d_state), dim=2)
        dts = F.conv1d(
            dts.reshape(batch, -1, length),
            self.dt_projs_weight.view(self.k_group * channels, -1, 1),
            groups=self.k_group)
        ys = selective_scan_fn(
            xs.view(batch, -1, length), dts,
            -self.A_logs.float().exp(), Bs.contiguous(), Cs.contiguous(),
            self.Ds.float(), self.dt_projs_bias.flatten().float(),
            backend=self.backend)
        merged = cross_merge_fn(
            ys.view(batch, self.k_group, channels, height, width),
            force_torch=self.backend == 'torch')
        return self.out_norm(merged.view(batch, channels, height, width)).to(x.dtype)

    def forward(self, x):
        x = self.act(self.conv2d(self.in_proj(x)))
        return self.dropout(self.out_proj(self.forward_core(x)))
