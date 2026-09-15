"""Run: python model/mamba_seg/tests/verify.py /path/to/VMamba.
Loads upstream AST definitions verbatim, replacing only unavailable kernel dispatch
with upstream's own PyTorch references. No upstream algorithm body is rewritten.
"""
import ast
import math
import sys
import tempfile
from pathlib import Path
from collections import OrderedDict
from functools import partial
from typing import Any, Optional, Callable
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, trunc_normal_
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mamba_seg.models.vmamba.backbone import VMamba
from mamba_seg.models.vmamba.ss2d import SS2D, cross_merge, cross_scan
from mamba_seg.ops.selective_scan import selective_scan_torch

torch.set_num_threads(2)
torch.manual_seed(17)
official = Path(sys.argv[1]) / 'classification/models'
namespace = dict(globals())
def definitions(filename, names=None):
    source = (official / filename).read_text()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and (names is None or node.name in names):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(official/filename), 'exec'), namespace)
definitions('csm_triton.py', {'cross_scan_fwd', 'cross_merge_fwd'})
definitions('csms6s.py', {'selective_scan_torch'})
namespace['cross_scan_fn'] = lambda x, **kw: namespace['cross_scan_fwd'](x)
namespace['cross_merge_fn'] = lambda x, **kw: namespace['cross_merge_fwd'](x)
namespace['selective_scan_fn'] = namespace['selective_scan_torch']
definitions('vmamba.py')

for h,w in [(3,5),(1,7),(6,4)]:
    x = torch.randn(2,3,h,w,requires_grad=True)
    torch.testing.assert_close(cross_scan(x), namespace['cross_scan_fwd'](x), rtol=0, atol=0)
    y = torch.randn(2,4,3,h*w,requires_grad=True)
    a = cross_merge(y,h,w)
    b = namespace['cross_merge_fwd'](y.reshape(2,4,3,h,w)).reshape(2,3,h,w)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(cross_merge(cross_scan(x),h,w),4*x)
    torch.autograd.gradcheck(lambda z: cross_merge(cross_scan(z),h,w), (x.double(),))
print('PASS directional values, rectangular/single-row cases and scan gradcheck')

ref = namespace['SS2D'](d_model=8,d_state=1,ssm_ratio=1.,d_conv=3,
    conv_bias=False,forward_type='v05_noz',channel_first=True)
new = SS2D(8,backend='torch')
new.load_state_dict(ref.state_dict(),strict=True)
x = torch.randn(2,8,3,5,requires_grad=True)
z = x.detach().clone().requires_grad_()
a,b = ref(x),new(z)
torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
probe = torch.randn_like(a)
(a*probe).sum().backward(); (b*probe).sum().backward()
torch.testing.assert_close(x.grad,z.grad,rtol=1e-4,atol=1e-6)
for (name,p),(_,q) in zip(sorted(ref.named_parameters()), sorted(new.named_parameters())):
    torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6,msg=name)
print('PASS SS2D upstream state keys, forward, input and all parameter gradients')

kwargs=dict(dims=96,depths=(2,2,8,2),ssm_d_state=1,ssm_ratio=1.,mlp_ratio=4.,drop_path_rate=.2)
ref = namespace['Backbone_VSSM'](**kwargs,ssm_conv_bias=False,forward_type='v05_noz',
    downsample_version='v3',patchembed_version='v2',norm_layer='ln2d').eval()
new = VMamba(**kwargs,backend='torch').eval()
new.load_state_dict(ref.state_dict(),strict=True)
with torch.no_grad():
    for shape in [(32,48),(31,35)]:
        x = torch.randn(1,3,*shape)
        a,b = ref(x),new(x)
        for p,q in zip(a,b): torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)
        print('PASS full Tiny features',shape,[tuple(t.shape) for t in b],
              'max_abs',max((p-q).abs().max().item() for p,q in zip(a,b)))
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp)/'weights.pth'
    state = {k:v for k,v in ref.state_dict().items() if not k.startswith('outnorm')}
    state['classifier.head.weight'] = torch.zeros(1000,768)
    torch.save({'model': state}, path)
    new.load_pretrained(path)
    torch.save({'state_dict': {'module.backbone.'+k:v for k,v in ref.state_dict().items()}},path)
    new.load_pretrained(path)
    state.pop('patch_embed.0.weight')
    torch.save({'model':state},path)
    try: new.load_pretrained(path)
    except RuntimeError: pass
    else: raise AssertionError('incomplete checkpoint was accepted')
print('PASS classification/segmentation checkpoint formats and incomplete-load rejection')
import runpy
config_path = str(Path(__file__).resolve().parents[1]/'configs/vmamba_tiny.py')
cfg = runpy.run_path(config_path)
assert cfg['model']['backbone']['depths'] == (2,2,8,2)
assert cfg['model']['decode_head']['num_classes'] == 150
assert cfg['optim_wrapper']['optimizer']['type'] == 'AdamW'
assert cfg['train_cfg']['max_iters'] == 160000
print('PASS standalone config values')
try:
    from mmengine.config import Config
except ImportError:
    print('SKIP MMEngine config parse: mmengine is not installed')
else:
    Config.fromfile(config_path, import_custom_modules=False)
    print('PASS MMEngine config parse')
