_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    '../yolox/yolox_tta.py'
]

img_scale = (640, 640)  # width, height
backend_args = None

data_root = '/home/akore/mmdet_project/dataset_occluvid_singleclass_40_treatable/'
dataset_type = 'BaseVideoDataset'
METAINFO = {
        'classes' : ('Occlusion', )
    }

img_norm_cfg = dict(
    mean=[0,0,0], std=[1023, 1023, 1023], to_rgb=True) # Single channel mean repeated for RGB with 10-bit normalization

# model settings
model = dict(
    type='YOLOXTSF',
        data_preprocessor=dict(
        type='DetDataPreprocessor'
        ),
        backbone=dict(
        type='CSPDarknet',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=1,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='DIoULoss',
            eps=1e-6,
            reduction='sum',
            loss_weight=2
            ),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    temporal_cfg=dict(
        type='TimeSformerTransformer',
        input_dim=128,  
        model_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1
    ),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.60), max_per_img=1))

train_pipeline = [
    dict(type='LoadMultiChannelImageFromFiles', color_type= 'grayscale', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='TemporalRandomFlip', prob=0.5),
    dict(type='ResizeSequences', scale=img_scale, keep_ratio=True),      
    dict(type='PadSequences', size_divisor=32, pad_val=114.0),
    dict(type='MultiFrameAddRGBChannel'),
    dict(type='TemporalNormalize', **img_norm_cfg),
    dict(type='FormatTemporalInput'),
    dict(type='PackTemporalInputs')
]

test_pipeline = [
    dict(type='LoadMultiChannelImageFromFiles', color_type= 'grayscale', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeSequences', scale=img_scale, keep_ratio=True),      
    dict(type='PadSequences', size_divisor=32, pad_val=114.0),
    dict(type='MultiFrameAddRGBChannel'),
    dict(type='TemporalNormalize', **img_norm_cfg),
    dict(type='FormatTemporalInput'),
    dict(type='PackTemporalInputs')
]

train_dataset = dict(
    type=dataset_type,  
    seq_len=3,
    data_root=data_root,
    metainfo=METAINFO,
    ann_file=data_root + 'annotations/train.json',
    data_prefix=dict(img_path=data_root + 'train/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=40),
    pipeline=train_pipeline
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    multiprocessing_context='spawn',
    dataset=train_dataset)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        seq_len=3,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file=data_root + 'annotations/val.json',
        data_prefix=dict(img_path=data_root + 'val/'),
        pipeline=test_pipeline,
        test_mode=True
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    #ann_file=data_root + 'annotations/val.json',
    metric='bbox',
    classwise=True,
    format_only=False,
    outfile_prefix='./work_dirs/yolox_tsf-occluvid-40bb-treatable-test/val'
)
test_evaluator = val_evaluator

# training settings
max_epochs = 20    
num_last_epochs = 2     
interval = 1      

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=interval)
val_cfg = dict(type='ValLoop')  
test_cfg = dict(type='TestLoop') 

# optimizer
# default 8 gpu
base_lr = 0.01
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.),
    clip_grad=dict(max_norm=10.0))

# learning rate
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=2,    
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=2,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='ConstantLR',
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=interval,
        max_keep_ckpts=30  
    ),
    visualization=dict(
        type='ModDetVisualizationHook', 
        #draw=True,
        score_thr=0.001,
        test_out_dir= './work_dirs/yolox_tsf-occluvid-40bb-treatable-test/results_test'
        ))

custom_hooks = [
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)


evaluation = dict(interval=1, metric='bbox', save_best='auto')

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')


load_from= 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
