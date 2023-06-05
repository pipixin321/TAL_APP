from operator import itemgetter
from itertools import groupby
import numpy as  np
import os
import json
import cv2

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

def show_in_video(cap,preds):
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration=cap.get(7)/cap.get(5)
    #resize param
    new_short=180
    if width>height:
        h=new_short
        w=int(width*h/height)
    else:
        w=new_short
        h=int(height*w/width)
    print(w,h)
    dw,dh=100,15
    out_mp4 = cv2.VideoWriter("./tmp/result.mp4", 
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              30,(w+dw,h+dh))
    ret=True
    while ret:
        t=cap.get(0)/1000
        ret,frame=cap.read()
        if ret:
            print(t)
            frame=cv2.resize(frame,(w,h))
            frame=cv2.putText(frame,"time:%.2fs"%(t),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)

            # initialize output frame
            color_lst=[[200,200,200],#mask两侧颜色
                    [255,255,255],#进度条底色
                    [255,0,0],#进度条背景片段颜色
                    [0,0,255]#进度条前景片段颜色
                    ]
            f_out=np.zeros((h+dh,w+dw),dtype=np.uint8)
            f_out=cv2.cvtColor(f_out,cv2.COLOR_GRAY2BGR)
            f_out[:,:,:]=color_lst[0]

            #frame
            f_out[:h,int(dw/2):int(w+dw/2),:]=frame

            #progress bar
            f_out[h:,int(dw/2):int(w+dw/2),:]=color_lst[1]

            #current timestamp
            cur_x=int(dw/2+w*t/duration)
            f_out[h:,int(dw/2):cur_x]=color_lst[2]

            #prediction
            #get prediction for current frame
            # max_score=0
            # pred_max=None
            # for pred in preds:
            #     score=pred['score']
            #     s,e=pred["segment"]
            #     if t>s and t<e and score>max_score:
            #         max_score=score
            #         pred_max=pred
            # if pred_max is not None:
            for pred in preds:
                score=pred['score']
                s,e=pred["segment"]
                if t>s and t<e:
                    f_out[:,0:int(dw/2)]=color_lst[3]
                    f_out[:,-int(dw/2):-1,:]=color_lst[3]
                    f_out[h:,int(dw/2+w*s/duration):cur_x]=color_lst[3]
                    f_out=cv2.putText(f_out,'prediction:{}({:.2f}s~{:.2f}s)'.format(pred["label"],s,e)
                                      ,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
                elif t>e:
                    f_out[h:,int(dw/2+w*s/duration):int(dw/2+w*e/duration)]=color_lst[3]

            out_mp4.write(f_out)
    cap.release()
    out_mp4.release()


if __name__ == '__main__':
    keep_result=post_process()
    print(keep_result)

    
    vid_dir='/mnt/data1/zhx/TAL_APP/tmp/video'
    vids=os.listdir(vid_dir)
    vid=vids[0]
    cap = cv2.VideoCapture(os.path.join(vid_dir,vid))
    show_in_video(cap,keep_result)
   
