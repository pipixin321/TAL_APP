import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from pprint import pprint


def generate_pgt(dataset_path,sample='gaussian'):
    print("sampling method:"+sample)
    gt_path=os.path.join(dataset_path,"gt_full.json")
    assert os.path.exists(gt_path)

    with open(gt_path,'r') as json_file:
        gt_dict=json.load(json_file)
    database=gt_dict['database']
    
    point_labels=pd.DataFrame(columns=['class','start_frame','stop_frame','video_id','point'])
    #calculate nums of instance
    total_num=0
    for video_id,annos in tqdm(database.items()):
        if annos["subset"] in ["Validation","train",'training']:
            fps=annos['fps']
            duration=annos['duration']

            for anno in annos['annotations']:
                total_num+=1
                start_frame,stop_frame=[fps*float(t) for t in anno['segment']]
                if sample=='gaussian':
                    alpha=np.random.normal(loc=0.5,scale=0.01)
                    alpha=min(max(alpha,0),1)
                    point_frame=int(start_frame+(stop_frame-start_frame)*alpha)
                elif sample=="uniform":
                    point_frame=int(np.random.uniform(start_frame,stop_frame))
                else:
                    print("unrecognized sample")
                    break
                
                f_len=25*duration//16
                p_pos=int(point_frame*25 / (fps * 16))
                if  p_pos> f_len:
                    print(p_pos,f_len,video_id)
                if stop_frame-start_frame>0 and p_pos<=f_len:#filter
                    temp_dict={}
                    temp_dict['class']=anno['label']
                    temp_dict['video_id']=video_id
                    temp_dict['start_frame']=start_frame
                    temp_dict['stop_frame']=stop_frame
                    temp_dict['point']=point_frame
                    point_labels.loc[len(point_labels)] = temp_dict
    pprint(point_labels)
    num_point_labels=len(point_labels)
    print(num_point_labels,total_num)

    save_path=os.path.join(dataset_path,'point_gaussian')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path=os.path.join(save_path,'point_labels.csv')
    point_labels.to_csv(file_path,index=False)
    print(file_path+" Saved!")
    

if __name__=="__main__":
    # dataset_path="/mnt/data1/zhx/dataset/THUMOS14"
    dataset_path="/root/Weakly_TAL/baseline_v1/dataset/GTEA"
    np.random.seed(0)
    generate_pgt(dataset_path)