import os
import numpy as np
import sys
sys.path.append('tal_alg/actionformer')

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data


# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed

thumos_class_list=['BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
                    'CliffDiving', 'CricketBowling', 'CricketShot', 'Diving',
                    'FrisbeeCatch', 'GolfSwing', 'HammerThrow', 'HighJump',
                    'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
                    'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking']

def infer_single_actionformer(backbone,vid,feats,fps,duration,gpu=0):
    topk = 100
    root = '/mnt/data1/zhx/TAL_APP/tal_alg/actionformer_release'
    config = os.path.join(root, 'configs/thumos_{}.yaml'.format(backbone)) 
    ckpt = os.path.join(root, 'ckpt/thumos_{}/best.pth.tar'.format(backbone))

    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    if os.path.isfile(config):
        cfg = load_config(config)
    cfg['model']['test_cfg']['max_seg_num'] = topk
    video_list = [
        {
            'video_id':vid.split('.')[0],
            'feats':torch.from_numpy(np.ascontiguousarray(feats.transpose())),
            'fps': fps,
            'duration':duration,
            'feat_stride':cfg['dataset']['feat_stride'],
            'feat_num_frames':cfg['dataset']['num_frames']
        }
    ]

    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=[device])
    model.device = device
    print("=> loading checkpoint '{}'".format(ckpt))
    checkpoint = torch.load(
        ckpt,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    print("Loading from EMA model ...")
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    model.eval()
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }
    
    with torch.no_grad():
        output = model(video_list)

        # unpack the results into ANet format
        num_vids = len(output)
        for vid_idx in range(num_vids):
            if output[vid_idx]['segments'].shape[0] > 0:
                results['video-id'].extend(
                    [output[vid_idx]['video_id']] *
                    output[vid_idx]['segments'].shape[0]
                )
                results['t-start'].append(output[vid_idx]['segments'][:, 0])
                results['t-end'].append(output[vid_idx]['segments'][:, 1])
                results['label'].append(output[vid_idx]['labels'])
                results['score'].append(output[vid_idx]['scores'])
    
    final_results = []
    for i in range(topk):
        temp_dict = {}
        temp_dict['label'] = thumos_class_list[results['label'][0][i]]
        temp_dict['score'] = float(results['score'][0][i].data.numpy())
        temp_dict['segment'] = [float(results['t-start'][0][i].data.numpy()),float(results['t-end'][0][i].data.numpy())]
        final_results.append(temp_dict)

    print('Done')
    return final_results

if __name__ == '__main__':
    import cv2
    vid_dir='/mnt/data1/zhx/TAL_APP/tmp/video'
    vids=os.listdir(vid_dir)
    vid=vids[0]
    cap = cv2.VideoCapture(os.path.join(vid_dir,vid))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    duration=frame_counter/fps

    import pickle
    tmp_feat_dir='/mnt/data1/zhx/TAL_APP/tmp/features'
    files=os.listdir(tmp_feat_dir)
    file=files[0]
    with open(os.path.join(tmp_feat_dir,file),'rb') as f:
        feats=pickle.load(f)
    

    infer_single_actionformer(vid,feats,fps,duration,gpu=0)