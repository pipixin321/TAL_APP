from operator import itemgetter
from itertools import groupby
import numpy as  np
import os
import json

def post_process(results=None,score_thresh=0.2,inter_thresh=0.9):
    tmp_json_path='./tmp/results.json'
    if results is not None and not os.path.exists(tmp_json_path):
        with open(tmp_json_path, 'w') as f:
            json.dump(results, f, indent=4)
    if results is None:
        with open(tmp_json_path,'r') as f:
            results=json.load(f)

    results=list(filter(lambda x:x['score']>score_thresh,results))

    keep_result=dict()
    for c,item in groupby(results, key=itemgetter('label')):
        result_c=list(item)
        keep_c=[result_c[0]]
        for i in range(1,len(result_c)):
            keep_c_sec=np.array([k['segment'] for k in keep_c])
            s,e=result_c[i]['segment']
            area=e-s
            xx1=np.maximum(s,keep_c_sec[:,0])
            xx2=np.minimum(e,keep_c_sec[:,1])
            inter = np.maximum(0.0, xx2 - xx1)
            
            if max(inter)==0:
                keep_c.append(result_c[i])
            elif max(inter)/area > inter_thresh:
                continue
            else:
                continue
        keep_result[c]=keep_c

    keep_result=[keep_result[c][i] for c in keep_result.keys() for i in range(len(keep_result[c]))]
    
    return keep_result

def show_in_video(cap,result):

    return None


if __name__ == '__main__':
    keep_result=post_process()
    print(keep_result)

    import cv2
    vid_dir='/mnt/data1/zhx/TAL_APP/tmp/video'
    vids=os.listdir(vid_dir)
    vid=vids[0]
    cap = cv2.VideoCapture(os.path.join(vid_dir,vid))
    cap = cv2.VideoCapture(vid_dir)