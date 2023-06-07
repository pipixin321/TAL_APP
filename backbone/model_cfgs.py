
MODEL_CFGS={
    'xxx':{
        'data_pipeline':[],
        'model_cfg':dict()
    },
    'i3d':{
        'data_pipeline':
            [
            dict(type='UntrimmedSampleFrames',clip_len=32,frame_interval=8,start_index=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 256)),
            dict(type='CenterCrop', crop_size=256),
            dict(type='Normalize',mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375],to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
            ],
        'model_cfg':
            dict(
            type='Recognizer3D',
            backbone=dict(
                type='ResNet3d',
                pretrained2d=False,
                pretrained='torchvision://resnet50',
                depth=50,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=2,
                pool1_stride_t=2,
                conv_cfg=dict(type='Conv3d'),
                norm_eval=False,
                inflate=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                zero_init_residual=False),
            cls_head=dict(
                type='I3DHead',
                num_classes=400,
                in_channels=2048,
                spatial_type='avg',
                dropout_ratio=0.5,
                init_std=0.01),
            test_cfg=dict(average_clips=None,feature_extraction=True)),
        'ckpt':'/mnt/data1/zhx/TAL_APP/backbone/ckpt/i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth'
    },

    'swin_tiny':{
        'data_pipeline':
            [
            dict(type='UntrimmedSampleFrames',clip_len=32,frame_interval=8,start_index=1),
            dict(type='RawFrameDecode'),
            dict(type='Resize', scale=(-1, 224)),
            dict(type='CenterCrop', crop_size=224),
            dict(type='Normalize',mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375],to_bgr=False),
            dict(type='FormatShape', input_format='NCTHW'),
            dict(type='Collect', keys=['imgs'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs'])
            ],
        'model_cfg':
            dict(
            type='Recognizer3D',
            backbone=dict(
                type='SwinTransformer3D',
                patch_size=(2,4,4),
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=(8,7,7),
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True),
            cls_head=dict(
                type='I3DHead',
                in_channels=768,
                num_classes=400,
                spatial_type='avg'),
            test_cfg=dict(average_clips=None,feature_extraction=True)),
        'ckpt':'/mnt/data1/zhx/TAL_APP/backbone/ckpt/swin_tiny_patch244_window877_kinetics400_1k.pth'
    }
}