# MambaSeg

用于遥感分割实验的 VMamba 模块。默认主干保留上游 v05_noz 的计算和权重名称。
新增的轻量金字塔解码器与 MMSeg UPerNet 是两条独立实验路线，效果需分别评估。

核心模型对齐仓库内官方 VMamba：保留 `Backbone_VSSM`、`VSSBlock`、`SS2D`、
`cross_scan_fn`、`cross_merge_fn` 和 `selective_scan_fn` 的命名与调用层次。
当前分割版本固定使用官方 `v2` patch embedding、`v3` downsample 和
`v05_noz` NCHW 路径；`VMamba` 是项目侧的兼容名称。

## 目录结构

```text
mamba_seg/
├── models/
│   ├── vmamba/
│   │   ├── ss2d.py
│   │   ├── vss_block.py
│   │   └── backbone.py
│   ├── decoder/
│   │   ├── decoder.py
│   │   └── upernet.py
│   └── segmentor.py
├── ops/
│   └── selective_scan/
│       ├── interface.py
│       ├── cuda/
│       │   ├── csrc/
│       │   └── setup.py
│       └── triton/
│           └── cross_scan.py
├── configs/
│   └── vmamba_tiny.py
├── datasets/
├── losses/
├── tests/
│   ├── test_model.py
│   └── verify.py
├── mamba_seg.py
├── train.py
└── test.py
```

| 文件                            | 职责 / 改进位置                          |
| ------------------------------- | ---------------------------------------- |
| mamba_seg.py                    | 主干 → 解码器 → 输入尺寸 logits        |
| models/decoder/decoder.py       | 原有轻量金字塔解码器                     |
| models/decoder/upernet.py       | UPerNet 解码器                           |
| models/vmamba/backbone.py       | 四阶段主干、特征元信息、预训练加载       |
| models/vmamba/vss_block.py      | VSS 残差块、NCHW 基础层和随机深度        |
| models/vmamba/ss2d.py           | 四向扫描、输入相关 Δ/B/C 与状态空间计算 |
| ops/selective_scan/interface.py | oflex CUDA / torch 参考后端的统一接口    |
| ops/selective_scan/cuda         | 官方 oflex CUDA 源码和构建脚本           |
| ops/selective_scan/triton       | 官方 VMamba Triton 四向扫描算子          |
| datasets、losses                | 项目数据集和损失函数扩展包               |
| train.py、test.py               | MMEngine/MMSeg 训练和评测入口            |
| models/segmentor.py             | 可选 MMSeg 注册入口                      |
| configs/vmamba_tiny.py          | ADE20K + UPerNet 基线                    |

## 快速使用

从项目根目录设置 `PYTHONPATH="$PWD/model"`，使用 `from mamba_seg import ...`。
这样不会触发顶层 `model/__init__.py` 对其他模型的导入。
基础入口只需 PyTorch；上游对照测试额外需要 timm。

```python
import torch
from mamba_seg import MambaSeg

# 小模型用于 CPU 检查；正式实验可使用默认 Tiny 宽度、深度和 oflex。
model = MambaSeg(
    in_chans=6, num_classes=2, decoder_channels=32,
    dims=16, depths=(1, 1, 1, 1), backend='torch',
)
logits = model(torch.randn(2, 6, 31, 45))  # [2, 2, 31, 45]
```

输出不含 softmax。多分类使用整数标签 [B,H,W] 和 CrossEntropyLoss；
单通道二分类设置 num_classes=1，配合同形状浮点标签和 BCEWithLogitsLoss。
模型不负责影像读取、归一化、NoData 掩膜或标签编码。

## 实验接口

- 多波段 / 光学与 SAR 通道拼接：设置 `in_chans`，输入需先配准并逐波段归一化。
- 默认解码器：`PyramidDecoder` 是原有轻量实现。
- UPerNet 解码器：使用 `UPerNetDecoder(backbone.out_channels, channels, num_classes)` 并通过 `decoder=` 传入。
- 自定义解码器：传 `decoder=模块`，接收特征列表，返回 NCHW 类别 logits。
- 自定义主干：传 `backbone=模块`，返回由细到粗的特征列表；默认解码器读取其 `out_channels`。
- 自定义残差块：通过 `block_cls` 和 `block_kwargs` 注入，构造参数遵循 VSSBlock。
- 自定义特征混合：传 `block_kwargs={'mixer_cls': MyMixer}`，构造参数遵循 SS2D，输入输出 NCHW 形状相同。
- 修改扫描方向：修改 ss2d.py 中的扫描函数和 `SS2D.forward_core`；改变方向数时需同步修改分组投影及参数形状。

`VMamba.out_channels` 和 `feature_strides` 对应所选 `out_indices`。
传入自定义 backbone 时，先自行配置并初始化权重，不再传额外 backbone_kwargs。
默认保留严格的主干权重检查；RGB 权重不能直接加载到多波段输入层，
需显式设计波段适配层或转换输入层权重。更换模块后也需检查权重兼容性。
