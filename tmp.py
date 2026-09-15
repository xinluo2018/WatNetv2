"""
vmamba_segmentation.py

Design goals
------------
1. Keep the current VMamba-Tiny segmentation backbone path easy to read.
2. Keep the four-direction SS2D idea explicit:
       Cross Scan -> Selective Scan -> Cross Merge
3. Return four hierarchical features:
       [B, 96, H/4,  W/4]
       [B,192, H/8,  W/8]
       [B,384, H/16, W/16]
       [B,768, H/32, W/32]
4. Add a lightweight FPN-style segmentation decoder so this file can be
   instantiated and run without MMSegmentation.

Default backbone configuration follows the current official VMamba-Tiny
downstream configuration:
    dims=96
    depths=(2, 2, 8, 2)
    d_state=1
    dt_rank="auto"
    ssm_ratio=1.0
    d_conv=3
    mlp_ratio=4.0
    drop_path_rate=0.2
    patch_embed="v2"
    downsample="v3"
    norm="ln2d"
    gate disabled (v05_noz-style path)

Notes
-----
- If mamba_ssm is installed, its optimized selective_scan_fn is used.
- Otherwise a slow PyTorch reference implementation is used. The reference
  implementation is useful for debugging/reading, not for serious training.
- This is a cleaned single-file research baseline, not a verbatim copy of
  every historical VMamba branch/backend.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Optional optimized selective scan
# ---------------------------------------------------------------------------

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _mamba_selective_scan_fn
except Exception:
    _mamba_selective_scan_fn = None


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------

class Linear2d(nn.Linear):
    """nn.Linear weights applied to BCHW data as a 1x1 convolution."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key = prefix + "weight"
        if key in state_dict:
            state_dict[key] = state_dict[key].view(self.weight.shape)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channel dimension for BCHW feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Stochastic depth."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """Channel-first MLP used by VSSBlock."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = Linear2d(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = Linear2d(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbedV2(nn.Module):
    """
    VMamba patch embedding v2 style:
    two 3x3 stride-2 convolutions -> total stride 4.
    """

    def __init__(
        self,
        in_chans: int = 3,
        embed_dim: int = 96,
        patch_norm: bool = True,
    ):
        super().__init__()
        mid_dim = embed_dim // 2

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_chans,
                mid_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(mid_dim) if patch_norm else nn.Identity(),
            nn.GELU(),
            nn.Conv2d(
                mid_dim,
                embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(embed_dim) if patch_norm else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class DownsampleV3(nn.Module):
    """
    VMamba v3-style stage downsampling:
        [B,C,H,W] -> [B,2C,H/2,W/2]
    """

    def __init__(self, dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or 2 * dim
        self.down = nn.Sequential(
            nn.Conv2d(
                dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=True,
            ),
            LayerNorm2d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


# ---------------------------------------------------------------------------
# Cross Scan / Cross Merge
# ---------------------------------------------------------------------------

def cross_scan(x: torch.Tensor) -> torch.Tensor:
    """
    Four-direction scan.

    Input
    -----
    x: [B, D, H, W]

    Output
    ------
    xs: [B, 4, D, L], L = H*W

    Directions
    ----------
    0: row-major
    1: column-major (implemented by transposing H/W first)
    2: reverse row-major
    3: reverse column-major
    """
    B, D, H, W = x.shape
    L = H * W

    row = x.flatten(2)
    col = x.transpose(2, 3).contiguous().flatten(2)

    xs = torch.stack(
        [
            row,
            col,
            torch.flip(row, dims=[-1]),
            torch.flip(col, dims=[-1]),
        ],
        dim=1,
    )
    assert xs.shape == (B, 4, D, L)
    return xs


def cross_merge(
    ys: torch.Tensor,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Restore four scan directions to normal image order and sum them.

    Input
    -----
    ys: [B, 4, D, L]

    Output
    ------
    y: [B, D, H, W]
    """
    B, K, D, L = ys.shape
    assert K == 4
    assert L == H * W

    y0 = ys[:, 0]

    y1 = (
        ys[:, 1]
        .reshape(B, D, W, H)
        .transpose(2, 3)
        .contiguous()
        .flatten(2)
    )

    y2 = torch.flip(ys[:, 2], dims=[-1])

    y3 = torch.flip(ys[:, 3], dims=[-1])
    y3 = (
        y3.reshape(B, D, W, H)
        .transpose(2, 3)
        .contiguous()
        .flatten(2)
    )

    y = y0 + y1 + y2 + y3
    return y.reshape(B, D, H, W)


# ---------------------------------------------------------------------------
# Selective Scan
# ---------------------------------------------------------------------------

def selective_scan_reference(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    Bv: torch.Tensor,
    Cv: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Slow readable PyTorch selective-scan reference.

    Shapes
    ------
    u, delta:  [B, D, L]
    A:         [D, N]
    Bv, Cv:    [B, G, N, L]
    D:         [D]
    delta_bias:[D]

    VMamba uses G=4 scan groups and D=4*d_inner.
    """
    batch, dim, length = u.shape
    state_dim = A.shape[-1]
    groups = Bv.shape[1]

    if dim % groups != 0:
        raise ValueError(f"dim={dim} must be divisible by groups={groups}")

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)

    if delta_softplus:
        delta = F.softplus(delta)

    channels_per_group = dim // groups

    # [B,G,N,L] -> [B,D,N,L]
    B_full = Bv.repeat_interleave(channels_per_group, dim=1)
    C_full = Cv.repeat_interleave(channels_per_group, dim=1)

    state = torch.zeros(
        batch,
        dim,
        state_dim,
        dtype=torch.float32,
        device=u.device,
    )
    outputs = []

    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_full = B_full.float()
    C_full = C_full.float()

    for t in range(length):
        dt = delta_f[:, :, t]                         # [B,D]
        u_t = u_f[:, :, t]                           # [B,D]

        dA = torch.exp(dt[:, :, None] * A_f[None])   # [B,D,N]
        dBu = (
            dt[:, :, None]
            * B_full[:, :, :, t]
            * u_t[:, :, None]
        )

        state = dA * state + dBu

        y_t = torch.sum(
            state * C_full[:, :, :, t],
            dim=-1,
        )

        if D is not None:
            y_t = y_t + D.float()[None, :] * u_t

        outputs.append(y_t)

    y = torch.stack(outputs, dim=-1)
    return y.to(dtype=u.dtype)


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    Bv: torch.Tensor,
    Cv: torch.Tensor,
    D: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """
    Use mamba_ssm's optimized kernel if available; otherwise reference code.
    """
    if _mamba_selective_scan_fn is not None:
        return _mamba_selective_scan_fn(
            u,
            delta,
            A,
            Bv,
            Cv,
            D,
            z=None,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=False,
        )

    return selective_scan_reference(
        u,
        delta,
        A,
        Bv,
        Cv,
        D=D,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
    )


# ---------------------------------------------------------------------------
# SS2D
# ---------------------------------------------------------------------------

class SS2D(nn.Module):
    """
    Clean v05_noz-style SS2D.

    Main path
    ---------
    BCHW
      -> in_proj
      -> depthwise conv
      -> SiLU
      -> Cross Scan
      -> project to delta/B/C
      -> Selective Scan
      -> Cross Merge
      -> norm
      -> out_proj
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 1,
        ssm_ratio: float = 1.0,
        dt_rank: Union[str, int] = "auto",
        d_conv: int = 3,
        conv_bias: bool = False,
        dropout: float = 0.0,
        bias: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = (
            math.ceil(d_model / 16)
            if dt_rank == "auto"
            else int(dt_rank)
        )
        self.k_group = 4

        self.in_proj = Linear2d(
            d_model,
            self.d_inner,
            bias=bias,
        )

        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=self.d_inner,
            bias=conv_bias,
        )

        self.act = nn.SiLU()

        # Per-direction projection:
        # D -> (dt_rank + d_state + d_state)
        x_proj = [
            nn.Linear(
                self.d_inner,
                self.dt_rank + 2 * self.d_state,
                bias=False,
            )
            for _ in range(self.k_group)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([m.weight for m in x_proj], dim=0)
        )

        # Low-rank delta projection: R -> D
        dt_projs = [
            self._dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
            )
            for _ in range(self.k_group)
        ]

        self.dt_projs_weight = nn.Parameter(
            torch.stack([m.weight for m in dt_projs], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([m.bias for m in dt_projs], dim=0)
        )

        self.A_logs = self._A_log_init(
            self.d_state,
            self.d_inner,
            copies=self.k_group,
            merge=True,
        )

        self.Ds = self._D_init(
            self.d_inner,
            copies=self.k_group,
            merge=True,
        )

        self.out_norm = LayerNorm2d(self.d_inner)

        self.out_proj = Linear2d(
            self.d_inner,
            d_model,
            bias=bias,
        )

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0.0
            else nn.Identity()
        )

    @staticmethod
    def _dt_init(
        dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
    ) -> nn.Linear:
        dt_proj = nn.Linear(
            dt_rank,
            d_inner,
            bias=True,
        )

        dt_init_std = dt_rank ** -0.5 * dt_scale

        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(
                dt_proj.weight,
                -dt_init_std,
                dt_init_std,
            )
        else:
            raise NotImplementedError(dt_init)

        dt = torch.exp(
            torch.rand(d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        # inverse softplus
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def _A_log_init(
        d_state: int,
        d_inner: int,
        copies: int = 1,
        merge: bool = True,
    ) -> nn.Parameter:
        A = torch.arange(
            1,
            d_state + 1,
            dtype=torch.float32,
        )
        A = A[None, :].repeat(d_inner, 1)
        A_log = torch.log(A)

        if copies > 1:
            A_log = A_log[None].repeat(copies, 1, 1)
            if merge:
                A_log = A_log.flatten(0, 1)

        return nn.Parameter(A_log)

    @staticmethod
    def _D_init(
        d_inner: int,
        copies: int = 1,
        merge: bool = True,
    ) -> nn.Parameter:
        D = torch.ones(d_inner)

        if copies > 1:
            D = D[None].repeat(copies, 1)
            if merge:
                D = D.flatten(0, 1)

        return nn.Parameter(D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        K = self.k_group
        D = self.d_inner
        N = self.d_state
        R = self.dt_rank
        L = H * W

        x = self.in_proj(x)
        x = self.conv2d(x)
        x = self.act(x)

        # [B,D,H,W] -> [B,K,D,L]
        xs = cross_scan(x)

        # [B,K,D,L] x [K,C,D] -> [B,K,C,L]
        x_dbl = torch.einsum(
            "bkdl,kcd->bkcl",
            xs,
            self.x_proj_weight,
        )

        dts, Bs, Cs = torch.split(
            x_dbl,
            [R, N, N],
            dim=2,
        )

        # [B,K,R,L] x [K,D,R] -> [B,K,D,L]
        dts = torch.einsum(
            "bkrl,kdr->bkdl",
            dts,
            self.dt_projs_weight,
        )

        xs_flat = xs.reshape(B, K * D, L)
        dts_flat = dts.reshape(B, K * D, L)

        As = -torch.exp(self.A_logs.float())
        Ds = self.Ds.float()
        delta_bias = self.dt_projs_bias.float().reshape(-1)

        ys = selective_scan(
            xs_flat,
            dts_flat,
            As,
            Bs,
            Cs,
            D=Ds,
            delta_bias=delta_bias,
            delta_softplus=True,
        )

        ys = ys.reshape(B, K, D, L)

        y = cross_merge(ys, H, W)
        y = self.out_norm(y)
        y = self.out_proj(y)
        y = self.dropout(y)

        return y


# ---------------------------------------------------------------------------
# VSS Block
# ---------------------------------------------------------------------------

class VSSBlock(nn.Module):
    """VMamba block = SS2D branch + MLP branch, each with residual."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 1,
        ssm_ratio: float = 1.0,
        dt_rank: Union[str, int] = "auto",
        d_conv: int = 3,
        conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.norm1 = LayerNorm2d(hidden_dim)

        self.op = SS2D(
            d_model=hidden_dim,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=drop,
        )

        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0.0

        if self.mlp_branch:
            self.norm2 = LayerNorm2d(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                out_features=hidden_dim,
                drop=drop,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(
            self.op(self.norm1(x))
        )

        if self.mlp_branch:
            x = x + self.drop_path(
                self.mlp(self.norm2(x))
            )

        return x


# ---------------------------------------------------------------------------
# VMamba Backbone
# ---------------------------------------------------------------------------

class VSSStage(nn.Module):
    """A stage contains several VSSBlocks, followed by optional downsampling."""

    def __init__(
        self,
        dim: int,
        depth: int,
        d_state: int,
        ssm_ratio: float,
        dt_rank: Union[str, int],
        d_conv: int,
        conv_bias: bool,
        mlp_ratio: float,
        drop: float,
        drop_path_rates: Sequence[float],
        downsample: bool = True,
        out_dim: int | None = None,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                VSSBlock(
                    hidden_dim=dim,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio,
                    dt_rank=dt_rank,
                    d_conv=d_conv,
                    conv_bias=conv_bias,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    drop_path=drop_path_rates[i],
                )
                for i in range(depth)
            ]
        )

        self.downsample = (
            DownsampleV3(dim, out_dim)
            if downsample
            else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.blocks(x)
        next_x = self.downsample(feature)
        return feature, next_x


class VMambaBackbone(nn.Module):
    """
    Four-stage VMamba backbone for dense prediction.

    Default = current Tiny downstream path:
        dims   = 96, 192, 384, 768
        depths = 2, 2, 8, 2
    """

    def __init__(
        self,
        in_chans: int = 3,
        dims: Union[int, Sequence[int]] = 96,
        depths: Sequence[int] = (2, 2, 8, 2),
        d_state: int = 1,
        dt_rank: Union[str, int] = "auto",
        ssm_ratio: float = 1.0,
        d_conv: int = 3,
        conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        out_indices: Sequence[int] = (0, 1, 2, 3),
        patch_norm: bool = True,
    ):
        super().__init__()

        if isinstance(dims, int):
            dims = [dims * (2 ** i) for i in range(4)]
        else:
            dims = list(dims)

        if len(dims) != 4:
            raise ValueError("dims must contain four stage dimensions")
        if len(depths) != 4:
            raise ValueError("depths must contain four stage depths")

        self.dims = dims
        self.depths = tuple(depths)
        self.out_indices = tuple(out_indices)

        self.patch_embed = PatchEmbedV2(
            in_chans=in_chans,
            embed_dim=dims[0],
            patch_norm=patch_norm,
        )

        total_depth = sum(depths)
        dpr = torch.linspace(
            0,
            drop_path_rate,
            total_depth,
        ).tolist()

        stages = []
        cursor = 0

        for i in range(4):
            stage = VSSStage(
                dim=dims[i],
                depth=depths[i],
                d_state=d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=dt_rank,
                d_conv=d_conv,
                conv_bias=conv_bias,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path_rates=dpr[cursor: cursor + depths[i]],
                downsample=(i < 3),
                out_dim=(dims[i + 1] if i < 3 else None),
            )
            stages.append(stage)
            cursor += depths[i]

        self.layers = nn.ModuleList(stages)

        for i in self.out_indices:
            self.add_module(
                f"outnorm{i}",
                LayerNorm2d(dims[i]),
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)

        outs = []

        for i, layer in enumerate(self.layers):
            feature, x = layer(x)

            if i in self.out_indices:
                norm = getattr(self, f"outnorm{i}")
                outs.append(norm(feature).contiguous())

        return outs

    def load_pretrained(
        self,
        checkpoint: str,
        key: str = "model",
        strict: bool = False,
    ):
        """
        Load a checkpoint with a tolerant policy.

        Because this file is a cleaned research implementation rather than
        the entire original source tree, inspect missing/unexpected keys.
        """
        ckpt = torch.load(
            checkpoint,
            map_location="cpu",
        )

        if isinstance(ckpt, dict) and key in ckpt:
            state_dict = ckpt[key]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Common wrappers
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ("module.", "backbone."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            cleaned[k] = v

        return self.load_state_dict(
            cleaned,
            strict=strict,
        )


# ---------------------------------------------------------------------------
# Lightweight dense prediction head
# ---------------------------------------------------------------------------

class FPNDecoder(nn.Module):
    """
    Simple FPN-style decoder.

    This is deliberately much smaller than copying MMSeg UPerNet into the
    same file. For official UPerNet experiments, use VMambaBackbone outputs
    with MMSegmentation's UPerHead.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        channels: int = 256,
        num_classes: int = 150,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.lateral = nn.ModuleList(
            [
                nn.Conv2d(c, channels, kernel_size=1)
                for c in in_channels
            ]
        )

        self.refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(
                channels * len(in_channels),
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.classifier = nn.Conv2d(
            channels,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(features) != 4:
            raise ValueError("FPNDecoder expects four backbone features")

        laterals = [
            conv(feat)
            for conv, feat in zip(self.lateral, features)
        ]

        # top-down fusion
        for i in range(3, 0, -1):
            laterals[i - 1] = (
                laterals[i - 1]
                + F.interpolate(
                    laterals[i],
                    size=laterals[i - 1].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )

        laterals = [
            refine(x)
            for refine, x in zip(self.refine, laterals)
        ]

        target_size = laterals[0].shape[-2:]

        fused = torch.cat(
            [
                x
                if x.shape[-2:] == target_size
                else F.interpolate(
                    x,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
                for x in laterals
            ],
            dim=1,
        )

        fused = self.fuse(fused)
        return self.classifier(fused)


# ---------------------------------------------------------------------------
# Complete semantic segmentation model
# ---------------------------------------------------------------------------

class VMambaSegmentation(nn.Module):
    """
    End-to-end semantic segmentation model:
        Image -> VMamba backbone -> FPN decoder -> logits
    """

    def __init__(
        self,
        num_classes: int,
        in_chans: int = 3,
        dims: int = 96,
        depths: Sequence[int] = (2, 2, 8, 2),
        d_state: int = 1,
        dt_rank: Union[str, int] = "auto",
        ssm_ratio: float = 1.0,
        d_conv: int = 3,
        conv_bias: bool = False,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.2,
        decoder_channels: int = 256,
    ):
        super().__init__()

        stage_dims = [
            dims,
            dims * 2,
            dims * 4,
            dims * 8,
        ]

        self.backbone = VMambaBackbone(
            in_chans=in_chans,
            dims=stage_dims,
            depths=depths,
            d_state=d_state,
            dt_rank=dt_rank,
            ssm_ratio=ssm_ratio,
            d_conv=d_conv,
            conv_bias=conv_bias,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            out_indices=(0, 1, 2, 3),
        )

        self.decode_head = FPNDecoder(
            in_channels=stage_dims,
            channels=decoder_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        features = self.backbone(x)
        logits = self.decode_head(features)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return logits


def vmamba_tiny_seg(
    num_classes: int = 150,
    **kwargs,
) -> VMambaSegmentation:
    """Factory for the current VMamba-Tiny-style segmentation baseline."""
    return VMambaSegmentation(
        num_classes=num_classes,
        dims=96,
        depths=(2, 2, 8, 2),
        d_state=1,
        dt_rank="auto",
        ssm_ratio=1.0,
        d_conv=3,
        conv_bias=False,
        mlp_ratio=4.0,
        drop_path_rate=0.2,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Quick check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # For a quick CPU syntax/shape smoke test without mamba_ssm, use a small
    # image because the reference selective scan is intentionally slow.
    model = vmamba_tiny_seg(num_classes=19)
    model.eval()

    x = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        features = model.backbone(x)
        print("Backbone feature shapes:")
        for i, f in enumerate(features):
            print(f"  F{i+1}: {tuple(f.shape)}")

        y = model(x)
        print("Segmentation logits:", tuple(y.shape))
