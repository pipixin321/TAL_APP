import os,time
from extract_rawframes import extract_from_video
from extract_features import set_model,extract_feat
from tal_alg.CoRL.inference import infer_single
from post_process import post_process
import pickle
import cv2
from termcolor import colored


def process_video(tmp_dir):
    vid_path=os.path.join(tmp_dir,'video')
    frame_path=os.path.join(tmp_dir,'rawframes')
    feat_path=os.path.join(tmp_dir,'features')

    FRAME_CFGS={
        'src_dir':vid_path,
        'out_dir':frame_path,
        'task':'rgb',
        'level':1,
        'ext':'mp4',
        'new_short':0,
        'use_opencv':True
    }
    extract_from_video(FRAME_CFGS)
    print('Finish extract raw frames')

    MODEL_CFGS={}
    data_pipeline,model=set_model(MODEL_CFGS)

    FEAT_CFGS={
        'data_prefix':os.path.join(tmp_dir,'rawframes'),
        'output_prefix':os.path.join(tmp_dir,'features'),
        'batch_size':1,
        'modality':'RGB'
    }
    extract_feat(data_pipeline,model,FEAT_CFGS)
    print('Finish extract feature')

    #tal
    print(colored('>>>Temporal Action Localizing...'))
    t1 = time.perf_counter()
    vid_dir=os.path.join(vid_path,os.listdir(vid_path)[0])
    cap = cv2.VideoCapture(vid_dir)
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    duration=frame_counter/fps
    feat_dir=os.path.join(feat_path,os.listdir(feat_path)[0])
    with open(feat_dir,'rb') as f:
        vid_feat=pickle.load(f)
    results=infer_single(vid_feat,duration)
    t2 = time.perf_counter()
    print(colored('<Localization Done>:','green')+'running time {:.3f} s'.format(t2-t1))

    keep_results=post_process(results,score_thresh=0.2)


    print(keep_results)
    return keep_results

if __name__ == '__main__':
    results=process_video('./tmp')
    