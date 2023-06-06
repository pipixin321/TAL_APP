import argparse
import os
import os.path as osp
import pickle,time

import mmcv
import numpy as np
import torch
from tqdm import tqdm
from termcolor import colored

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Extract Feature')
    parser.add_argument('--data-prefix', default='', help='input_data')
    parser.add_argument('--output-prefix', default='', help='output prefix')
    parser.add_argument('--data-list',default='',help='video list of the dataset')
    # parser.add_argument(
    #     '--frame-interval',
    #     type=int,default=16,
    #     help='the sampling frequency of frame in the untrimed video')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'Flow'])
    parser.add_argument('--ckpt', help='checkpoint for feature extraction')
    parser.add_argument(
        '--part',type=int,default=0,
        help='which part of dataset to forward(alldata[part::total])')
    parser.add_argument('--total', type=int, default=1, help='how many parts exist')
    parser.add_argument('--resume', action='store_true', default=True, help='resume')
    parser.add_argument('--max-frame', type=int, default=15000, help='max number of video frames')
    args = parser.parse_args()

    return args

def set_model(MODEL_CFGS):
    #swin_tiny
    data_pipeline=[
        dict(type='UntrimmedSampleFrames',clip_len=32,frame_interval=8,start_index=1),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, 224)),
        dict(type='CenterCrop', crop_size=224),
        dict(type='Normalize',mean=[123.675, 116.28, 103.53],std=[58.395, 57.12, 57.375],to_bgr=False),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    model_cfg = dict(
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
    test_cfg=dict(average_clips=None,feature_extraction=True))
    ckpt='/mnt/data1/zhx/TAL_APP/backbone/swinvivit/swin_tiny_patch244_window877_kinetics400_1k.pth'

    #load model
    t0 =time.perf_counter() 
    model = build_model(model_cfg)
    model.load_state_dict(torch.load(ckpt)['state_dict'])
    t1 = time.perf_counter()
    print(colored('<Model Built>:','green')+'running time {:.3f} s'.format(t1-t0))

    return data_pipeline,model


# def extract_feat(data_pipeline,model,tmp_path):
def extract_feat(data_pipeline,model,FEAT_CFGS):
    args = parse_args()
    args.data_prefix=FEAT_CFGS['data_prefix']
    args.output_prefix=FEAT_CFGS['output_prefix']
    args.batch_size=FEAT_CFGS['batch_size']
    args.modality=FEAT_CFGS['modality']
    if not osp.exists(args.output_prefix):
        os.makedirs(args.output_prefix)
    
    args.is_rgb = args.modality == 'RGB'
    args.f_tmpl = 'img_{:05d}.jpg' if args.is_rgb else 'flow_{}_{:05d}.jpg'

    #load to gpu device
    t1 = time.perf_counter()
    gpu_id=args.part
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    t2 = time.perf_counter()
    print(colored('<Model Loaded>:','green')+'device={},running time {:.3f} s'.format(device,t2-t1))
    model.eval()

    #get data list
    data=sorted(os.listdir(args.data_prefix))
    finished_lst=os.listdir(args.output_prefix) if args.resume else []
    finished_lst=[f.split('.')[0] for f in finished_lst]
    print(colored('<Date Collected>:','green')+'{} videos total,{} done'.format(len(data),len(finished_lst)))

    if osp.exists('./blocked_videos.txt'):
        with open('./blocked_videos.txt') as f:
            blocked_videos=[x.strip() for x in f.readlines()]
            finished_lst+=blocked_videos

    data = data[args.part::args.total]
    prog_bar = mmcv.ProgressBar(len(data))
    for item in data:
        if item in finished_lst:
            prog_bar.update()
            continue
        #filter out too short and too long videos
        frame_dir=osp.join(args.data_prefix,item)
        num_frames=len(os.listdir(frame_dir))
        if num_frames<32:
            print(colored('\n<Warning>:','red')+'video is too short,{}:{}(>32frames)'.format(item,num_frames))
            prog_bar.update()
            continue
        
        # prepare sample
        print(colored('\n>>>Clipping Video...','yellow')+'{}:{} frames'.format(item,num_frames))
        if num_frames<args.max_frame:
            video_feat=forward_data(args,frame_dir,num_frames,1,
                                    data_pipeline,model,device,args.batch_size)
            output_file = osp.join(args.output_prefix, osp.basename(item)+'.pkl')
            with open(output_file, 'wb') as fout:
                pickle.dump(video_feat, fout)
                print(colored('<Feature Saved>:','green')+item+'.pkl')
            prog_bar.update()
        else:
            print(colored('\n<Warning>:','red')+'video is too long,{}:{}(<{}frames)'.format(item,num_frames,args.max_frame))
            video_feats=[]
            nums_divide=num_frames//args.max_frame+1
            for n in range(nums_divide-1):
                video_feat=forward_data(args,frame_dir,args.max_frame,n*args.max_frame+1,
                                        data_pipeline,model,device,args.batch_size)
                video_feats.append(video_feat)

            if num_frames%args.max_frame > 32:
                video_feat=forward_data(args,frame_dir,num_frames%args.max_frame,(n+1)*args.max_frame+1,
                                        data_pipeline,model,device,args.batch_size)
                video_feats.append(video_feat)
            
            video_feats=np.concatenate(video_feats)
            output_file = osp.join(args.output_prefix, osp.basename(item)+'.pkl')
            with open(output_file, 'wb') as fout:
                pickle.dump(video_feats, fout)
                print(colored('<Feature Saved>:','green')+item+'.pkl'+' vidfeat shape={}'.format(video_feats.shape))
            prog_bar.update()

def forward_data(args,frame_dir,num_frames,start_index,
                 data_pipeline,model,device,batch_size):
    iters_data=tqdm(range(1),desc='Clipping Video...')
    for i in iters_data:
        t3 = time.perf_counter()
        tmpl = dict(frame_dir=frame_dir,total_frames=num_frames,
                    filename_tmpl=args.f_tmpl,start_index=start_index,modality=args.modality)
        sample = data_pipeline(tmpl)
        imgs = sample['imgs'] #[N,C,T,H,W]
        imgs = imgs.unsqueeze(1)
        t4 = time.perf_counter()
        print(colored('<Video Clipped>:','green')+'clip shape={},running time {:.3f} s'.format(imgs.shape,t4-t3))

    print(colored('>>>Extracting Feature...','yellow'))
    results=[]
    num_clip = imgs.shape[0]
    total_iters=num_clip//batch_size
    with torch.no_grad():
        iters_feat=tqdm(range(total_iters),total=total_iters,desc='Extracting Feature') #if progress is None else progress.tqdm(range(total_iters),desc='Extracting Feature')
        for i in iters_feat:
            part=imgs[batch_size*i:batch_size*(i+1)]
            part = part.to(device)
            feat = model.forward(part, return_loss=False)
            results.append(feat)
        if num_clip%batch_size!=0:
            part=imgs[batch_size*(i+1):]
            part = part.to(device)
            feat = model.forward(part, return_loss=False)
            results.append(feat)
    results = np.concatenate(results)
    t5 = time.perf_counter()
    print(colored('<Feature Extracted>:','green')+'feat shape={},running time {:.3f} s'.format(results.shape,t5-t4))

    return results


# if __name__ == '__main__':
#     MODEL_CFGS={}
#     data_pipeline,model,device=set_model(MODEL_CFGS)

#     tmp_dir='./tmp'
#     FEAT_CFGS={
#         'data_prefix':os.path.join(tmp_dir,'rawframes'),
#         'output_prefix':os.path.join(tmp_dir,'features'),
#         'batch_size':5,
#         'modality':'RGB'
#     }
#     extract_feat(data_pipeline,model,device,FEAT_CFGS)
#     print('Finish extract feature')


#model_config 
#data_config:clip_len  interval
   