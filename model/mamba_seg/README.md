# MambaSeg

用于遥感语义分割的 VMamba 实现。当前主要学习和实验入口是 `mamba_seg.py`，
该文件整合了基础层、SS2D、VSSBlock、四阶段主干和轻量解码器。

## 目录与入口

```text
mamba_seg/
├── mamba_seg.py             # 单文件模型：MambaSeg、VMamba_Backbone、PyramidDecoder
├── tmp.ipynb                # 模型与权重加载实验
├── models/                 # 保留的模块化实现，与单文件实现独立
│   ├── vmamba/              # backbone.py、vss_block.py、ss2d.py
│   ├── decoder/             # 轻量解码器及自定义 UPerNetDecoder
│   └── segmentor.py         # MMSeg 注册入口
├── ops/selective_scan/
│   ├── interface.py         # torch / oflex 统一调用与自动求导
│   ├── cuda/                # oflex 扩展源码及 setup.py
│   └── triton/              # GPU 四方向展开与合并
├── configs-tmp/             # 保留的 MMSeg 配置
├── tests-del/               # 历史验证脚本
├── datasets-del/            # 保留目录
├── losses-del/              # 保留目录
├── mamba_seg-del.py          # 历史代码
├── train.py                 # MMSeg 配置训练入口
├── test.py                  # MMSeg 配置评测入口
├── VERIFICATION.txt         # 历史验证记录
└── LICENSE
```

## 运行流程

```text
输入 [B,C,H,W]
  → Patch Embedding（两次 stride=2 卷积）
  → 四阶段 VMamba（VSSBlock + 阶段间下采样）
  → 多尺度输出（归一化后送入解码器）
  → PyramidDecoder（通道投影、由深到浅上采样相加、类别预测）
  → 上采样至原图尺寸
  → logits [B,num_classes,H,W]
```

VSSBlock 包含 `LN2D → SS2D → DropPath → 残差相加` 和
`LN2D → MLP → DropPath → 残差相加` 两个分支。
SS2D 将特征沿行、列及两个反向展开，由输入生成 Δ/B/C，
执行状态递推，再对齐并求和四方向输出。Δ 的偏置和 Softplus 在扫描接口中处理。

默认 Tiny 主干为 `dims=96`、`depths=(2,2,8,2)`、`ssm_ratio=1.0`、
`ssm_d_state=1`，四阶段通道为 [96,192,384,768]。
固定结构为 v05_noz、LN2D、patch embed v2、downsample v3；
相关版本参数用于拒绝不支持的配置，不会动态切换网络结构。
Identity 占位层保留官方 checkpoint 的层索引。

## 快速使用

在项目根目录运行，Notebook 也需让项目根目录位于 Python 搜索路径中。
明确导入单文件中的类：

```python
import torch
from model.mamba_seg.mamba_seg import MambaSeg

# 小模型、CPU 参考后端，用于检查形状和数据流程。
model = MambaSeg(
    in_chans=6, num_classes=1, decoder_channels=32,
    dims=16, depths=(1, 1, 1, 1), backend="torch",
)
logits = model(torch.randn(2, 6, 31, 45))  # [2,1,31,45]
```

模型主体使用 PyTorch；上述导入同时经过项目 `model/__init__.py`，
因此也可能加载其他模型的依赖。GPU 四方向算子使用 Triton，oflex 另需编译 CUDA 扩展。
torch 参考后端仅适合小图调试，大图会占用较多内存并显著变慢。

多分类使用整数标签 [B,H,W] 和 CrossEntropyLoss；
单通道二分类使用同形状浮点标签和 BCEWithLogitsLoss。
模型输出没有 Sigmoid/Softmax，训练时直接使用 logits。
输入波段应预先配准、归一化，并处理 NoData 和标签编码。

## 加载 Tiny 预训练权重

现有权重位于项目的 `model/weights/vssm1_tiny_0230s_ckpt_epoch_264.pth`，
对应 Tiny s1l8。在默认主干创建时传入 pretrained，会自动加载主干；
轻量解码器仍随机初始化。

```python
model = MambaSeg(
    in_chans=6, num_classes=1, decoder_channels=32,
    dims=96, depths=(2, 2, 8, 2), ssm_ratio=1.0, ssm_d_state=1,
    backend="oflex",
    pretrained="/home/xin/Developer-luo/WatNetv2/model/weights/"
               "vssm1_tiny_0230s_ckpt_epoch_264.pth",
)
model = model.to("cuda")
```

单文件 `VMamba_Backbone.load_pretrained()` 已针对 RGB → 六波段做适配：

```python
key = "patch_embed.0.weight"
if key in state:
    weight = state[key]
    if weight.shape[1] == 3 and self.patch_embed[0].in_channels == 6:
        state[key] = weight.repeat(1, 2, 1, 1) * 0.5
```

这是输入层的初始化策略，不表示新增波段获得了预训练语义。
适配仅覆盖 3 → 6 通道，其余结构不匹配仍会报错。
修改源码后需重新导入并创建模型；Notebook 可重启内核后重新运行。
检查 `model.backbone.patch_embed[0].in_channels` 应与实际输入通道一致。

## 后端与改进接口

oflex 编译需要与 PyTorch CUDA 主版本兼容的 Toolkit、开发头文件及 C++ 编译器；
本地构建脚本还会查询 GPU 算力。用实际运行模型的 Python 环境安装并验证：

```bash
python -m pip install ninja
python -m pip install --no-build-isolation ./model/mamba_seg/ops/selective_scan/cuda
python -c "import torch, selective_scan_cuda_oflex; print(torch.cuda.is_available()); print(selective_scan_cuda_oflex.__file__)"
```

- `in_chans`、`num_classes`：设置波段数和类别数。
- `decoder=模块`：接收特征列表，返回 NCHW logits；最终尺寸由 MambaSeg 恢复。
- `backbone=模块`：自行构建并初始化主干；默认解码器需要其 out_channels。
- `block_cls` / `block_kwargs`：替换残差块或传入 mixer_cls；遵循现有构造接口。
- `use_checkpoint=True`：以反向时重算部分特征换取更少显存。
- 修改扫描方向时，同步调整分组投影与参数形状。

传入已创建的 backbone 后，不再传 dims、depths、pretrained 等 backbone_kwargs。
自定义主干的预训练加载需显式调用 `backbone.init_weights()` 或相应加载接口。

## MMSeg 配置路线

train.py / test.py 读取 `configs-tmp/vmamba_tiny.py`，
通过 models/segmentor.py 注册模块化主干，并使用 MMSeg 的 UPerHead 和辅助 FCNHead。
这条路线不调用单文件的 MambaSeg，单文件的六通道适配也不会自动应用。

从项目根目录执行：

```bash
python model/mamba_seg/train.py model/mamba_seg/configs-tmp/vmamba_tiny.py --work-dir work_dirs/vmamba
python model/mamba_seg/test.py model/mamba_seg/configs-tmp/vmamba_tiny.py /path/to/checkpoint.pth --work-dir work_dirs/vmamba_test
```

需要兼容的 MMEngine、MMSeg、MMCV 和扫描后端。配置仍为 ADE20K 基线，
遥感应用需同步调整数据集、路径、类别、标签约定、读取器和各波段 mean/std；
多波段数据需调整 RGB 通道转换和颜色增强。
models/decoder/upernet.py 是项目自定义 UPerNet 解码器，
不能假定与官方 MMSeg UPerHead 的权重完全兼容。

VERIFICATION.txt 和 tests-del 为历史验证记录，不代表当前单文件或当前环境已重新验证。
代码参考 MzeroMiko/VMamba，许可证见 LICENSE。
