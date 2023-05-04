norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained=
    'pretrained/segformer_mit-b1_512x512_160k_ade20k_20210726_112106-d70e859d.pth',
    backbone=dict(type='mit_b1', style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=10,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'RARP50Dataset'
data_root = '/media/deep/Transcend/sar-rarp-dataset/traindata'
img_norm_cfg = dict(
    mean=[60.99, 26.03, 26.96], std=[43.87, 30.58, 33.15], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[60.99, 26.03, 26.96],
        std=[43.87, 30.58, 33.15],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(
                type='Normalize',
                mean=[60.99, 26.03, 26.96],
                std=[43.87, 30.58, 33.15],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/traindata',
        img_dir=[
            'video_1.zip/rgb', 'video_2.zip/rgb', 'video_3.zip/rgb',
            'video_4.zip/rgb', 'video_5.zip/rgb', 'video_6.zip/rgb',
            'video_7.zip/rgb', 'video_8.zip/rgb', 'video_9.zip/rgb',
            'video_10.zip/rgb', 'video_11.zip/rgb', 'video_12.zip/rgb',
            'video_13.zip/rgb', 'video_14.zip/rgb', 'video_15.zip/rgb',
            'video_16.zip/rgb', 'video_17.zip/rgb', 'video_18.zip/rgb',
            'video_19.zip/rgb', 'video_20.zip/rgb', 'video_21.zip/rgb',
            'video_22.zip/rgb', 'video_23.zip/rgb', 'video_24.zip/rgb',
            'video_25.zip/rgb', 'video_26.zip/rgb', 'video_27.zip/rgb',
            'video_28.zip/rgb', 'video_29.zip/rgb', 'video_30.zip/rgb',
            'video_31.zip/rgb', 'video_32.zip/rgb', 'video_33.zip/rgb',
            'video_34.zip/rgb', 'video_35.zip/rgb', 'video_36.zip/rgb',
            'video_37.zip/rgb', 'video_38.zip/rgb', 'video_39.zip/rgb',
            'video_40.zip/rgb'
        ],
        ann_dir=[
            'video_1.zip/segmentation', 'video_2.zip/segmentation',
            'video_3.zip/segmentation', 'video_4.zip/segmentation',
            'video_5.zip/segmentation', 'video_6.zip/segmentation',
            'video_7.zip/segmentation', 'video_8.zip/segmentation',
            'video_9.zip/segmentation', 'video_10.zip/segmentation',
            'video_11.zip/segmentation', 'video_12.zip/segmentation',
            'video_13.zip/segmentation', 'video_14.zip/segmentation',
            'video_15.zip/segmentation', 'video_16.zip/segmentation',
            'video_17.zip/segmentation', 'video_18.zip/segmentation',
            'video_19.zip/segmentation', 'video_20.zip/segmentation',
            'video_21.zip/segmentation', 'video_22.zip/segmentation',
            'video_23.zip/segmentation', 'video_24.zip/segmentation',
            'video_25.zip/segmentation', 'video_26.zip/segmentation',
            'video_27.zip/segmentation', 'video_28.zip/segmentation',
            'video_29.zip/segmentation', 'video_30.zip/segmentation',
            'video_31.zip/segmentation', 'video_32.zip/segmentation',
            'video_33.zip/segmentation', 'video_34.zip/segmentation',
            'video_35.zip/segmentation', 'video_36.zip/segmentation',
            'video_37.zip/segmentation', 'video_38.zip/segmentation',
            'video_39.zip/segmentation', 'video_40.zip/segmentation'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(
                type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(
                type='Normalize',
                mean=[60.99, 26.03, 26.96],
                std=[43.87, 30.58, 33.15],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/traindata',
        img_dir='video_val_1.zip/rgb',
        ann_dir='video_val_1.zip/segmentation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1080),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[60.99, 26.03, 26.96],
                        std=[43.87, 30.58, 33.15],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RARP50Dataset',
        data_root='/media/deep/Transcend/sar-rarp-dataset/traindata',
        img_dir='video_test_1.zip/rgb',
        ann_dir='video_test_1.zip/segmentation',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1920, 1080),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(
                        type='Normalize',
                        mean=[60.99, 26.03, 26.96],
                        std=[43.87, 30.58, 33.15],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU')
work_dir = './work_dirs/segformer.b1.512x512.rarp50.80k'
gpu_ids = range(0, 1)
