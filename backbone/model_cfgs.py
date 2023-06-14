import os
ckpt_path='/mnt/data1/zhx/TAL_APP/backbone/ckpt'
MODEL_CFGS={
    'xxx':{
        'data_pipeline':[],
        'model_cfg':dict(),
        'ckpt':''
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
        'ckpt':os.path.join(ckpt_path,'swin_tiny_patch244_window877_kinetics400_1k.pth')
    },

    'swin_base':{
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
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
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
                in_channels=1024,
                num_classes=600,
                spatial_type='avg'),
            test_cfg=dict(average_clips=None,feature_extraction=True)),
        'ckpt':os.path.join(ckpt_path,'swin_base_patch244_window877_kinetics600_22k.pth')
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
                pretrained2d=True,
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
        'ckpt':os.path.join(ckpt_path,'i3d_r50_32x2x1_100e_kinetics400_rgb_20200614-c25ef9a4.pth')
    },

    'slowfast101':{
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
                type='ResNet3dSlowFast',
                pretrained=None,
                resample_rate=4,  # tau
                speed_ratio=4,  # alpha
                channel_ratio=8,  # beta_inv
                slow_pathway=dict(
                    type='resnet3d',
                    depth=101,
                    pretrained=None,
                    lateral=True,
                    fusion_kernel=7,
                    conv1_kernel=(1, 7, 7),
                    dilations=(1, 1, 1, 1),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    inflate=(0, 0, 1, 1),
                    norm_eval=False),
                fast_pathway=dict(
                    type='resnet3d',
                    depth=101,
                    pretrained=None,
                    lateral=False,
                    base_channels=8,
                    conv1_kernel=(5, 7, 7),
                    conv1_stride_t=1,
                    pool1_stride_t=1,
                    norm_eval=False)),
            cls_head=dict(
                type='SlowFastHead',
                in_channels=2304,  # 2048+256
                num_classes=400,
                spatial_type='avg'),
            test_cfg=dict(average_clips=None,feature_extraction=True)),
        'ckpt':os.path.join(ckpt_path,'slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth')
    },

    'csn':{
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
                type='ResNet3dCSN',
                pretrained2d=False,
                pretrained=  # noqa: E251
                'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',  # noqa: E501
                depth=152,
                with_pool2=False,
                bottleneck_mode='ir',
                norm_eval=True,
                bn_frozen=True,
                zero_init_residual=False),
            cls_head=dict(
                type='I3DHead',
                num_classes=400,
                in_channels=2048,
                spatial_type='avg'),
            test_cfg=dict(average_clips=None,feature_extraction=True)),

        'ckpt':os.path.join(ckpt_path,'ircsn_ig65m_pretrained_bnfrozen_r152_32x2x1_58e_kinetics400_rgb_20200812-9037a758.pth')
    },
}