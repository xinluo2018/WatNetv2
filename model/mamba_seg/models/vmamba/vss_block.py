# Adapted from MzeroMiko/VMamba (MIT); see LICENSE.
"""NCHW layers and the VSS residual block used by this backbone."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


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


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = Linear2d(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = Linear2d(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class VSSBlock(nn.Module):
    """Official pre-norm SS2D and MLP residual structure."""
    def __init__(self, dim, drop_path=0.,
                        mlp_ratio=4., 
                        use_checkpoint=False,
                        ssm_d_state=1, 
                        ssm_ratio=1., 
                        ssm_dt_rank='auto',
                        backend='oflex', 
                        mixer_cls=None):
        super().__init__()
        if mixer_cls is None:
            from .ss2d import SS2D
            mixer_cls = SS2D
        self.norm = LayerNorm2d(dim)
        self.op = mixer_cls(
            dim, ssm_d_state, ssm_ratio, ssm_dt_rank, backend=backend)
        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.use_checkpoint = use_checkpoint

    def _forward(self, x):
        x = x + self.drop_path(self.op(self.norm(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)
