from operator import itemgetter
from itertools import groupby
import numpy as  np
import os
import json
import cv2

def post_process(result_file, score_thresh=0.2, inter_thresh=0.9):
    # tmp_json_path='./tmp/results.json'
    # if results is not None and not os.path.exists(tmp_json_path):
    #     with open(tmp_json_path, 'w') as f:
    #         json.dump(results, f, indent=4)
    # if results is None:
    #     with open(tmp_json_path,'r') as f:
    #         results=json.load(f)
    with open(result_file, 'r') as f:
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
    keep_result.sort(key=lambda k: k['segment'][0])
    
    return keep_result

def show_in_video(vid_dir, preds):
    cls_lst=[]
    for p in preds:
        if p['label'] not in cls_lst:
            cls_lst.append(p['label'])
    
    cap = cv2.VideoCapture(vid_dir)
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
    dw,dh=100,10
    out_mp4 = cv2.VideoWriter("./tmp/result.mp4", 
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              30,(w+dw,h+dh))
    ret=True
    while ret:
        t=cap.get(0)/1000
        ret,frame=cap.read()
        if ret:
            # print(t)
            frame=cv2.resize(frame,(w,h))
            frame=cv2.putText(frame,"time:%.2fs"%(t),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)

            # initialize output frame
            color_lst=[[255,255,255],#进度条底色
                       [200,200,200],#mask两侧颜色
                        [153,153,153],#进度条背景片段颜色
                        [[153,0,255],[0,102,255],[255,0,102],[102,255,0],[0,255,255],[0,0,255],[255,102,255],
                         [153,0,255],[0,102,255],[255,0,102],[102,255,0],[0,255,255],[0,0,255],[255,102,255]]#进度条前景片段颜色
                        ]
            f_out=np.zeros((h+dh,w+dw),dtype=np.uint8)
            f_out=cv2.cvtColor(f_out,cv2.COLOR_GRAY2BGR)
            f_out[:,:,:]=color_lst[0]

            #progress bar
            f_out[h:,int(dw/2):int(w+dw/2),:]=color_lst[1]

            #current timestamp
            cur_x=int(dw/2+w*t/duration)
            f_out[h:,int(dw/2):cur_x]=color_lst[2]

            #prediction
            #get prediction for current frame
            max_score=0
            cur_pred=None
            for pred in preds:
                if t>pred["segment"][0] and t<pred["segment"][1] and pred['score']>max_score:
                    max_score=pred['score']
                    cur_pred=pred
            past_pred=[]
            for pred in preds:
                if pred['segment'][1] < t:
                    past_pred.append(pred)
            
            if cur_pred is not None:
                s,e=cur_pred["segment"]
                f_color=color_lst[3][cls_lst.index(cur_pred['label'])]
                f_out[:,0:int(dw/2)-5]=f_color
                f_out[:,-int(dw/2)+5:-1,:]=f_color
                f_out[h:,int(dw/2+w*s/duration):cur_x]=f_color
                frame=cv2.putText(frame,'prediction:{}({:.2f}s~{:.2f}s)'.format(cur_pred["label"],s,e)
                                      ,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,255,0),1)
            if len(past_pred)>0:
                for p_pred in past_pred:
                    s,e=p_pred["segment"]
                    p_color=color_lst[3][cls_lst.index(p_pred['label'])]
                    f_out[h:,int(dw/2+w*s/duration):int(dw/2+w*e/duration)]=p_color
    
            #frame
            f_out[:h,int(dw/2):int(w+dw/2),:]=frame

            out_mp4.write(f_out)
    cap.release()
    out_mp4.release()

def trim_video(vid_dir, preds):
    print(preds)

    cap = cv2.VideoCapture(vid_dir)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    duration=cap.get(7)/cap.get(5)
    out_mp4 = cv2.VideoWriter("./tmp/result_trimmed.mp4", 
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              30,(int(width),int(height)))

    ret=True
    while ret:
        t=cap.get(0)/1000
        ret, frame=cap.read()
        if ret:
            for pred in preds:
                if t>pred["segment"][0] and t<pred["segment"][1]:
                    out_mp4.write(frame)
                    break
    cap.release()
    out_mp4.release()


if __name__ == '__main__':
    keep_result=post_process('/mnt/data1/zhx/TAL_APP/tmp/results_swin_tiny_ActionFormer(Fully-supervised).json')
    print(keep_result)

    # show_in_video('/mnt/data1/zhx/TAL_APP/tmp/video/video_test_0000062.mp4',keep_result)
    trim_video('/mnt/data1/zhx/TAL_APP/tmp/video/video_test_0000062.mp4',keep_result)
   
