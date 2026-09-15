# Adapted from MzeroMiko/VMamba (MIT); see LICENSE.
"""Standalone ADE20K baseline. Dataset-specific settings are grouped below."""
custom_imports = dict(imports=['mamba_seg.models.segmentor'], allow_failed_imports=False)
default_scope = 'mmseg'

# Change these together when adapting the dataset.
num_classes = 150
in_chans = 3
crop_size = (512, 512)
dataset_type = 'ADE20KDataset'
data_root = 'data/ade/ADEChallengeData2016'
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor', 
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375], 
    bgr_to_rgb=True,
    pad_val=0, 
    seg_pad_val=255, 
    size=crop_size)

model = dict(
    type='EncoderDecoder', data_preprocessor=data_preprocessor, pretrained=None,
    backbone=dict(
        type='MM_VMamba', dims=96, depths=(2, 2, 8, 2), in_chans=in_chans,
        out_indices=(0, 1, 2, 3), ssm_d_state=1, ssm_ratio=1.,
        ssm_dt_rank='auto', mlp_ratio=4., drop_path_rate=.2,
        backend='oflex', pretrained=None),
    decode_head=dict(
        type='UPerHead', in_channels=[96, 192, 384, 768], in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6), channels=512, dropout_ratio=.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
    auxiliary_head=dict(
        type='FCNHead', in_channels=384, in_index=2, channels=256,
        num_convs=1, concat_input=False, dropout_ratio=.1,
        num_classes=num_classes, norm_cfg=norm_cfg, align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=.4)),
    train_cfg={}, test_cfg=dict(mode='whole'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='RandomResize', scale=(2048, 512), ratio_range=(.5, 2.), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=.75),
    dict(type='RandomFlip', prob=.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')]
img_ratios = [.5, .75, 1., 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='TestTimeAug', transforms=[
        [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
        [dict(type='RandomFlip', prob=p, direction='horizontal') for p in (0., 1.)],
        [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]])]

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root,
                 data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                 pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type, data_root=data_root,
                 data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
                 pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(.9, .999), weight_decay=.01),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'relative_position_bias_table': dict(decay_mult=0.),
        'norm': dict(decay_mult=0.)}))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(type='PolyLR', eta_min=0., power=1., begin=1500, end=160000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
env_cfg = dict(cudnn_benchmark=True,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
               dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
