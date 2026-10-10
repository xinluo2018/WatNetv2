# Selective scan operators

`interface.py` provides the backend dispatch used by SS2D and a slow PyTorch
reference implementation for CPU checks.

- `cuda/` contains the official VMamba oflex CUDA source and build script.
- `triton/cross_scan.py` contains the official VMamba Triton cross-scan kernels.

Build the CUDA extension from `cuda/` in the target Python environment:

```bash
cd ops/selective_scan/cuda
pip install -v .
```

The resulting `selective_scan_cuda_oflex` module must be importable by Python.
