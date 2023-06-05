import os
import json
import numpy as np
from tqdm import tqdm
from eval.eval_detection import ANETdetection
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')


class_dict = {0: 'BaseballPitch',
                1: 'BasketballDunk',
                2: 'Billiards',
                3: 'CleanAndJerk',
                4: 'CliffDiving',
                5: 'CricketBowling',
                6: 'CricketShot',
                7: 'Diving',
                8: 'FrisbeeCatch',
                9: 'GolfSwing',
                10: 'HammerThrow',
                11: 'HighJump',
                12: 'JavelinThrow',
                13: 'LongJump',
                14: 'PoleVault',
                15: 'Shotput',
                16: 'SoccerPenalty',
                17: 'TennisSwing',
                18: 'ThrowDiscus',
                19: 'VolleyballSpiking'}
class_name_to_idx = dict((v, k) for k, v in class_dict.items()) 

def eval_pseudo_label(root_dir,steps):

    data_path='/root/Weakly_TAL/dataset/thumos14'
    with open(os.path.join(data_path,"gt_full.json")) as gt_f:
        gt_full=json.load(gt_f)
        gt_dict=gt_full["database"]

    data_list = [item.strip() for item in list(open(os.path.join(data_path, "split_train.txt")))]
    output_dir=os.path.join(root_dir,'pseudo_label')


    mAP_dict={}
    mAP_save_dir=os.path.join(root_dir,'pseudo_mAP.json')
    for step in steps:
        pseudo_label=np.load(os.path.join(output_dir,'act_step{}.npy'.format(step)),allow_pickle=True)

        result_dict={'version':"14",'results':{},'external_data':''}
        for i in tqdm(range(len(data_list))):
            vid_name=data_list[i]
            vid_pseudo_label=pseudo_label[i]

            loc_gt=gt_dict[vid_name]
            t_factor=vid_pseudo_label.shape[0]/loc_gt['duration']

            gt_class=[]
            for gt in loc_gt['annotations']:
                if gt['label'] not in gt_class:
                    gt_class.append(gt['label'])
            gt_class_idx=[class_name_to_idx[cls_name] for cls_name in gt_class]

            #genarate pseudo label
            annotations=[]
            vid_pseudo_label=vid_pseudo_label[:,gt_class_idx]

            for c in range(len(gt_class_idx)):
                psd_label=vid_pseudo_label[:,c]
                arr= np.where(psd_label>0)[0]
                psd_loc_labels=np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)
                for loc in psd_loc_labels:
                    anno={}
                    if len(loc)==1:
                        s,e=loc[0],min(loc[0]+1,psd_label.shape[0])
                    else:
                        s,e=loc[0],loc[-1]
                    s,e=round(s/t_factor,1),round(e/t_factor,1)
                    anno['label']=gt_class[c]
                    anno['segment']=[s,e]
                    anno['segment(frames)']=[int(s*loc_gt['fps']),int(e*loc_gt['fps'])]
                    anno['label_id']=gt_class_idx[c]
                    anno['score']=1.0

                    annotations.append(anno)

            result_dict['results'][vid_name]=annotations

        json_path = os.path.join(root_dir, 'temp_pseudo_result.json')
        with open(json_path, 'w') as f:
            json.dump(result_dict, f)

        gt_path=os.path.join(data_path,"gt_full.json")
        anet_detection = ANETdetection(gt_path, json_path,
                                        subset='Validation', tiou_thresholds=np.linspace(0.1, 0.7, 7),
                                        verbose=False, check_status=False)
        mAP, _ = anet_detection.evaluate()
        mean_mAP=mAP[:7].mean()
        
        print(step,mean_mAP)

        mAP_dict[str(step)]=list(mAP)+[mean_mAP]
        save_dict=json.dumps(mAP_dict,indent=4)
        with open(mAP_save_dir,'w') as json_file:
            json_file.write(save_dict)

def plot_contrast_curve(root_list):
    if not  isinstance(root_list,list):
        root_list=[root_list]

    gs = gridspec.GridSpec(6, 18,wspace=0.5,hspace=0.3)
    fig=plt.figure()
    mean_ax=fig.add_subplot(gs[:,:6])
    ax_list=[]
    for i1 in range(4):
        ax=fig.add_subplot(gs[:3,6+3*i1:6+3*(i1+1)])
        ax_list.append(ax)
    for i2 in range(3):
        ax=fig.add_subplot(gs[3:,6+3*i2:6+3*(i2+1)])
        ax_list.append(ax)

    
    colors=['red','green','orange','purple']
    for idx,root in enumerate(root_list):
        color=colors[idx%len(colors)]
        mAP_save_dir=os.path.join(root,'pseudo_mAP.json')
        with open(mAP_save_dir,'r') as json_file:
            mAP_dict=json.load(json_file)

        # for k,v in mAP_dict.items():
        x=[float(m) for m in mAP_dict.keys()]
        for i in range(7):
            y=[mAP[i] for mAP in mAP_dict.values()]
            ax_list[i].plot(x,y,linestyle='-',marker="o",markersize=4,label=root.split('/')[-1],color=color)
            
        mean_y=[mAP[-1] for mAP in mAP_dict.values()]
        mean_ax.plot(x,mean_y,linestyle='-',marker="o",markersize=4,label=root.split('/')[-1],color=color)
        
    
    fig.set_size_inches((20,10))
    mean_ax.grid(True)
    for i in range(7):
        ax_list[i].grid(True)
    mean_ax.legend()
    plt.savefig(os.path.join(root_list[0],'step_pseudo_mAP.png'),dpi=300)

        

if __name__ == "__main__":
    root='/root/Weakly_TAL/baseline/ckpt/THUMOS14/outputs/SA_pseudo'
    run_infos=['agnostic_supervised_step10','agnostic_supervised','agnostic']
    steps=np.arange(50,1000,50)
    # run_infos=['lacp']
    # steps=[1300]
    
    for run_info in run_infos:
        root_dir=os.path.join(root,run_info)
        eval_pseudo_label(root_dir,steps)

    root_dir_list=[os.path.join(root,run_info) for run_info in run_infos]
    plot_contrast_curve(root_dir_list)

