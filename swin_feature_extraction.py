# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import pickle

import mmcv
import numpy as np
import torch

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Extract TSN Feature')
    parser.add_argument('--data-prefix', default='/root/mmaction2/data/ActivityNet/rawframes', help='dataset prefix')
    parser.add_argument('--output-prefix', default='/root/ActivityNet2022/dataset/features/swinvivit_898', help='output prefix')
    parser.add_argument('--data-list',help='video list of the dataset, the format should be ''`frame_dir num_frames output_file`')
    parser.add_argument('--frame-interval',type=int,default=8,help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--ckpt',default="/root/ActivityNet2022/framework/Video-Swin-Transformer/anet_finetune/work_dir2/best_top1_acc_epoch_25.pth", help='checkpoint for feature extraction')
    # parser.add_argument(
    #     '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='which part of dataset to forward(alldata[part::total])')
    parser.add_argument(
        '--total', type=int, default=4, help='how many parts exist')
    args = parser.parse_args()
    return args


def main(args):
    os.environ['CUDA_VIVIBLE_DEVICES'] = str(args.part)
    device = torch.device("cuda:{}".format(args.part) if torch.cuda.is_available() else "cpu")
    args.device = device

    args.is_rgb = args.modality == 'RGB'
    args.clip_len = 32 if args.is_rgb else 5
    args.input_format = 'NCTHW' if args.is_rgb else 'NCHW_Flow'
    rgb_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False)
    flow_norm_cfg = dict(mean=[128, 128], std=[128, 128])
    args.img_norm_cfg = rgb_norm_cfg if args.is_rgb else flow_norm_cfg
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'
    args.in_channels = args.clip_len * (3 if args.is_rgb else 2)
    # max batch_size for one forward
    args.batch_size = 1

    # define the data pipeline for Untrimmed Videos
    data_pipeline = [
    dict(type='UntrimmedSampleFrames', clip_len=args.clip_len, frame_interval=args.frame_interval, start_index=0),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=256),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format=args.input_format),
    dict(type='Collect', keys=['imgs'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
    data_pipeline = Compose(data_pipeline)

    # define TSN R50 model, the model is used as the feature extractor
    # model_cfg = dict(
    # type='Recognizer3D',
    # backbone=dict(
    #     type='ResNet3dCSN',
    #     pretrained2d=False,
    #     pretrained=
    #     'https://download.openmmlab.com/mmaction/recognition/csn/ircsn_from_scratch_r152_ig65m_20200807-771c4135.pth',
    #     depth=152,
    #     with_pool2=False,
    #     bottleneck_mode='ir',
    #     norm_eval=True,
    #     zero_init_residual=False,
    #     bn_frozen=True),
    # cls_head=dict(
    #     type='I3DHead',
    #     num_classes=200,
    #     in_channels=2048,
    #     spatial_type='avg',
    #     dropout_ratio=0.5,
    #     init_std=0.01),
    # train_cfg=None,
    # test_cfg=dict(average_clips=None))
    model_cfg = dict(
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
        num_classes=200,
        spatial_type='avg',
        dropout_ratio=0.5),
    # test_cfg = dict(average_clips='prob'))
    test_cfg = dict(average_clips=None,feature_extraction=True))


    model = build_model(model_cfg)
    # load pretrained weight into the feature extractor
    # state_dict = torch.load(args.ckpt)['state_dict']
    state_dict = torch.load(args.ckpt,map_location=torch.device("cpu"))['state_dict']
    model.load_state_dict(state_dict)
    model=model.to(args.device)
    model.eval()

    # data = open(args.data_list).readlines()
    # data = [x.strip() for x in data]
    # data=os.listdir(args.data_prefix)
    data=open("/root/ActivityNet2022/framework/Video-Swin-Transformer/tools/data/activitynet/leftdata_list.txt").readlines()
    data = [x.strip() for x in data]
    print(len(data))
    data = data[args.part::args.total]

    # enumerate Untrimmed videos, extract feature from each of them
    prog_bar = mmcv.ProgressBar(len(data))
    if not osp.exists(args.output_prefix):
        os.system(f'mkdir -p {args.output_prefix}')

    def forward_data(model, data):
            # chop large data into pieces and extract feature from them
            results = []
            start_idx = 0
            num_clip = data.shape[0]
            while start_idx < num_clip:
                with torch.no_grad():
                    part = data[start_idx:start_idx + args.batch_size]
                    feat = model.forward(part, return_loss=False)
                    results.append(feat)
                    start_idx += args.batch_size
            return np.concatenate(results)
    
    max_length=2000
    for item in data:
        # frame_dir, length, _ = item.split()
        frame_dir=item
        output_file = osp.basename(frame_dir) + '.pkl'
        frame_dir = osp.join(args.data_prefix, frame_dir)
        output_file = osp.join(args.output_prefix, output_file)
        assert output_file.endswith('.pkl')

        length=len(os.listdir(os.path.join(args.data_prefix,frame_dir)))
        if length<args.clip_len:
            print("Ignore Warning:{}:{}<{}".format(item,length,args.clip_len))
            video_feat=np.zeros((1,1024))
            with open(output_file, 'wb') as fout:
                pickle.dump(video_feat, fout)
            prog_bar.update()
            continue
        print("\nvideo:{},frame:{}".format(item,length))
        

        if os.path.exists(output_file):
            prog_bar.update()
            continue
        else:
            length = int(length)

            # prepare a pseudo sample
            tmpl = dict(
                frame_dir=frame_dir,
                total_frames=length,
                filename_tmpl=args.f_tmpl,
                start_index=0,
                modality=args.modality)
            sample = data_pipeline(tmpl)
            imgs = sample['imgs']
            shape = imgs.shape
            # the original shape should be N_seg * C * H * W, resize it to N_seg *
            # 1 * C * H * W so that the network return feature of each frame (No
            # score average among segments)
            # print(shape)
            imgs = imgs.reshape((shape[0], 1) + shape[1:])
            # print(imgs.shape)
            if length>max_length:
                print("Video is too large,cut to frags")
                
                n_clip=imgs.shape[0]
                # frags=16
                frags=n_clip//50+1
                print(n_clip)
                video_feat=None
                for i in range(frags-1):
                    frag_data=imgs[n_clip//frags*i:n_clip//frags*(i+1)]
                    frag_data=frag_data.to(args.device)
                    frag_feat=forward_data(model, frag_data)
                    print(frag_feat.shape)
                    if video_feat is None:
                        video_feat=frag_feat
                    else:
                        video_feat=np.concatenate((video_feat,frag_feat),0)
                frag_data=imgs[n_clip//frags*(frags-1):]
                frag_data=frag_data.to(args.device)
                frag_feat=forward_data(model, frag_data)
                print(frag_feat.shape)

                video_feat=np.concatenate((video_feat,frag_feat),0)
                print(video_feat.shape)
            else:
                imgs=imgs.to(args.device)
                video_feat=forward_data(model, imgs)

            with open(output_file, 'wb') as fout:
                pickle.dump(video_feat, fout)
            prog_bar.update()


if __name__ == '__main__':
    args = parse_args()
    try:
        main(args)
    except:
        os.system("bash /root/ActivityNet2022/framework/Video-Swin-Transformer/tools/data/activitynet/swin_feat{}.sh".format(args.part))

