import os,time
from extract_rawframes import extract_from_video
from extract_features import set_model,extract_feat
from tal_alg.CoRL.inference import infer_single_CoRL
from tal_alg.actionformer_release.inference import infer_single_actionformer
from post_process import post_process,show_in_video
import pickle
import cv2
from termcolor import colored
from tqdm import tqdm
from backbone.model_cfgs import MODEL_CFGS


def process_video(tmp_dir,new_short,backbone,detector):
    #initialize path
    vid_path=os.path.join(tmp_dir,'video')
    frame_path=os.path.join(tmp_dir,'rawframes')
    feat_path=os.path.join(tmp_dir,'features')
    backbone_gpu=0
    detector_gpu=0

    #extract rawframes
    for i in tqdm(range(1),total=1,desc='Extracting rawframes...'):
        FRAME_CFGS={
            'src_dir':vid_path,
            'out_dir':frame_path,
            'task':'rgb',
            'level':1,
            'ext':'mp4',
            'new_short':new_short,
            'use_opencv':True
        }
        extract_from_video(FRAME_CFGS)
        print('Finish extract raw frames')

    #extract feature
    backbone_trans={'I3D':'i3d','SlowFast':'slowfast101','CSN':'csn','SwinViViT':'swin_tiny'}
    if detector == 'CoRL(Weakly-Supervised)': backbone = 'I3D'
    backbone=backbone_trans[backbone]
    data_pipeline,model=set_model(MODEL_CFGS[backbone])

    FEAT_CFGS={
        'data_prefix':os.path.join(tmp_dir,'rawframes'),
        'output_prefix':os.path.join(tmp_dir,'features'),
        'batch_size':1,
        'modality':'RGB'
    }
    extract_feat(data_pipeline,model,FEAT_CFGS,backbone_gpu)
    print('Finish extract feature')

    #temporal action localization
    for i in tqdm(range(1),total=1,desc='Localizing...'):
        print(colored('>>>Temporal Action Localizing...','yellow'))
        t1 = time.perf_counter()
        vid_name = os.listdir(vid_path)[0]
        vid_dir=os.path.join(vid_path,vid_name)
        cap = cv2.VideoCapture(vid_dir)
        frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        duration=frame_counter/fps
        feat_dir=os.path.join(feat_path,os.listdir(feat_path)[0])
        with open(feat_dir,'rb') as f:
            vid_feat=pickle.load(f)

        
        if detector == 'CoRL(Weakly-Supervised)':
            results=infer_single_CoRL(backbone,vid_feat,duration,gpu=detector_gpu)
        elif detector == 'ActionFormer(Fully-supervised)':
            results = infer_single_actionformer(backbone,vid_name,vid_feat,fps,duration,gpu=detector_gpu)
        else:
            print("Unvalid Detector")
        t2 = time.perf_counter()
        print(colored('<Localization Done>:','green')+'running time {:.3f} s'.format(t2-t1))

        print(colored('>>>Post processing...','yellow'))
        t1 = time.perf_counter()
        keep_results=post_process(results,score_thresh=0.2)
        show_in_video(cap,keep_results)
        t2 = time.perf_counter()
        print(colored('<Post processing Done>:','green')+'running time {:.3f} s'.format(t2-t1))

    print(keep_results)
    return keep_results

if __name__ == '__main__':
    results=process_video('./tmp',
                          new_short=180,
                          backbone='I3D',
                          detector='ActionFormer(Fully-supervised)',
                        #   detector='CoRL(Weakly-Supervised)',
                          ) 
    