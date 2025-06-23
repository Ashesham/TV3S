_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/cityscapes_1024x1024_repeat_clips1.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder_clips',
    pretrained='/home/data_/workdir/2024/May_VideoSegmentMamba/VSFM/pretrained/swin_small_patch4_window7_224_22k.pth',
    backbone=dict(
        type='d2_Swins',
        style='pytorch'),
    decode_head=dict(
        type='VSFMHead_shift_city',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256, window_w=20, window_h=20, shift_size=10, real_shift=True, model_type=0),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        num_clips=4,
        ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


# data
data = dict(samples_per_gpu=1)   # original is 1
evaluation = dict(interval=160000, metric='mIoU')
