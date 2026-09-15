# Adapted from MzeroMiko/VMamba (MIT); see LICENSE and README.md.
"""VMamba selective-scan interface with the official oflex ABI."""
import torch


def selective_scan_torch(
    u: torch.Tensor,  # (B, K * C, L)
    delta: torch.Tensor,  # (B, K * C, L)
    A: torch.Tensor,  # (K * C, N)
    B: torch.Tensor,  # (B, K, N, L)
    C: torch.Tensor,  # (B, K, N, L)
    D: torch.Tensor = None,  # (K * C)
    delta_bias: torch.Tensor = None,  # (K * C)
    delta_softplus=True,
    oflex=True,
    *args,
    **kwargs,
):
    """Official PyTorch reference implementation used for CPU verification."""
    dtype_in = u.dtype
    Batch, K, N, L = B.shape
    KCdim = u.shape[1]
    Cdim = int(KCdim / K)
    assert u.shape == (Batch, KCdim, L)
    assert delta.shape == (Batch, KCdim, L)
    assert A.shape == (KCdim, N)
    assert C.shape == B.shape

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)

    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    B = B.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    C = C.view(Batch, K, 1, N, L).repeat(1, 1, Cdim, 1, 1).view(Batch, KCdim, N, L)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    x = A.new_zeros((Batch, KCdim, N))
    ys = []
    for i in range(L):
        x = deltaA[:, :, i, :] * x + deltaB_u[:, :, i, :]
        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)
    y = torch.stack(ys, dim=2)

    out = y if D is None else y + u * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


class SelectiveScanCuda(torch.autograd.Function):
    """Autograd wrapper matching VMamba's official oflex extension ABI."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None,
                delta_softplus=False, oflex=True, backend=None):
        if backend != 'oflex':
            raise ValueError(f'Unsupported CUDA selective scan backend: {backend}')
        try:
            import selective_scan_cuda_oflex
        except ImportError as exc:
            raise RuntimeError(
                'Build ops/selective_scan/cuda first; see its README.md') from exc

        ctx.delta_softplus = delta_softplus
        ctx.backend = backend
        u, delta, B, C = [tensor.contiguous() for tensor in (u, delta, B, C)]
        out, x, *rest = selective_scan_cuda_oflex.fwd(
            u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        import selective_scan_cuda_oflex

        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1)
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None


def selective_scan_fn(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    delta_bias: torch.Tensor = None,
    delta_softplus=True,
    oflex=True,
    backend=None,
):
    """Dispatch with the same call signature as official VMamba ``csms6s.py``."""
    if backend == 'torch':
        fn = selective_scan_torch
    elif backend in (None, 'oflex'):
        fn = SelectiveScanCuda.apply
        backend = 'oflex'
    else:
        raise ValueError(f'Unknown selective scan backend: {backend}')
    return fn(u, delta, A, B, C, D, delta_bias, delta_softplus, oflex, backend)


# Compatibility with the project's earlier public name.
selective_scan = selective_scan_fn
