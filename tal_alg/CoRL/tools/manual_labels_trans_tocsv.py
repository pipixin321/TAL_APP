import pandas as pd
import json
import os
from tqdm import tqdm

with open("/mnt/data1/zhx/dataset/THUMOS14/gt_full.json","r") as f:
    gt_dict=json.load(f)
database=gt_dict['database']

def txt2csv(manual_id=1):
    data_path="/mnt/data1/zhx/Weakly_TAL/SF-Net/data/Thumos14-Annotations/single_frames/THUMOS{}.txt".format(manual_id)
    manual_points=pd.read_csv(data_path, sep=",",names=['video_id','point','class'])
    num_point=len(manual_points)
    print(num_point)

    save_num=0
    match_num=0
    point_labels=pd.DataFrame(columns=['class','start_frame','stop_frame','video_id','point'])
    for i in tqdm(range(num_point)):
        temp_dict={}
        cls=manual_points.iloc[i,:]['class'].strip()
        video_id=manual_points.iloc[i,:]['video_id']
        t_point=manual_points.iloc[i,:]['point']
        #match gt
        video_gts=database[video_id]['annotations']
        match_flag=0
        for gt in video_gts:
            if t_point>=float(gt['segment'][0]) and t_point<=float(gt['segment'][1]):
                match_flag=1
                t_start=float(gt['segment'][0])
                t_end=float(gt['segment'][1])
                
                fps=database[video_id]['fps']
                temp_dict['class']=cls
                temp_dict['video_id']=video_id
                temp_dict['point']=t_point*fps
                temp_dict['start_frame']=t_start*fps
                temp_dict['stop_frame']=t_end*fps
                point_labels.loc[len(point_labels)] = temp_dict
                save_num+=1
                break

        if match_flag==0:
            min_d=1e10
            for gt in video_gts:
                d=min(abs(t_point-float(gt['segment'][0])),abs(t_point-float(gt['segment'][0])))
                if d<min_d:
                    min_d=d
                    t_start=float(gt['segment'][0])
                    t_end=float(gt['segment'][1])
            print(t_point,t_start,t_end)
            fps=database[video_id]['fps']
            temp_dict['class']=cls
            temp_dict['video_id']=video_id
            temp_dict['point']=t_point*fps
            temp_dict['start_frame']=t_start*fps
            temp_dict['stop_frame']=t_end*fps
            point_labels.loc[len(point_labels)] = temp_dict
            match_num+=1
            
    point_labels.to_csv(os.path.join("/mnt/data1/zhx/dataset/THUMOS14/point_labels","point_labels_manual{}.csv".format(manual_id)),index=False)
    print("in_num:{},match_num:{},total:{}".format(save_num,match_num,save_num+match_num))

txt2csv(3)
# for id in [1,2,3,4]:
#     txt2csv(id)
    
    