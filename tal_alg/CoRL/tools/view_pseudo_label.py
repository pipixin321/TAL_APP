import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',family='Times New Roman') 
from pylab import mpl
mpl.rcParams['font.size'] = 12

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

def visual_pseudo_label(root_dir,steps,n):
    data_path='./dataset/THUMOS14'
    with open(os.path.join(data_path,"gt_full.json")) as gt_f:
        gt_dict=json.load(gt_f)["database"]
    point_anno = pd.read_csv(os.path.join(data_path, 'point_labels', 'point_labels_gaussian.csv'))
    data_list = [item.strip() for item in list(open(os.path.join(data_path, "split_train.txt")))]
    np.random.seed(0)
    select_idx=np.random.choice(len(data_list),n)
    # data_list=np.array(data_list)[select_idx]


    output_dir=os.path.join(root_dir,'pseudo_label')
    cas_lst,act_lst,bkg_lst,con_lst=[],[],[],[]
    matchid_lst,weight_lst=[],[]

    
    for step in steps:
        # cas=np.load(os.path.join(output_dir,'cas_step{}.npy'.format(step)),allow_pickle=True)
        # cas_lst.append(cas)
        act=np.load(os.path.join(output_dir,'act_step{}.npy'.format(step)),allow_pickle=True)
        act_lst.append(act)
        bkg=np.load(os.path.join(output_dir,'bkg_step{}.npy'.format(step)),allow_pickle=True)
        bkg_lst.append(bkg)
        con=np.load(os.path.join(output_dir,'con_step{}.npy'.format(step)),allow_pickle=True)
        con_lst.append(con)

        matchid=np.load(os.path.join(output_dir,'matchid_step{}.npy'.format(step)),allow_pickle=True)
        matchid_lst.append(matchid)
        weight=np.load(os.path.join(output_dir,'weight_step{}.npy'.format(step)),allow_pickle=True)
        weight_lst.append(weight)
        
    save_path=os.path.join(root_dir,'pseudo_label_view')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in tqdm(select_idx):
        vid_name=data_list[i]
        #load pseudo label
        # cas=[cas[i] for cas in cas_lst]
        act_labels=[act[i] for act in act_lst]
        bkg_labels=[bkg[i] for bkg in bkg_lst]
        con_labels=[con[i] for con in con_lst]
        matchids=[matchid[i] for matchid in matchid_lst]
        weights=[weight[i] for weight in weight_lst]
        T=act_labels[0].shape[0]

        c=con_labels[-1]
        num_cons=c.max(1).sum()

        #load gt
        loc_gt=gt_dict[vid_name]
        t_factor=act_labels[0].shape[0]/loc_gt['duration']

        box_gt={}#sort by class
        for gt in loc_gt['annotations']:
            if gt["label"] not in box_gt.keys():
                box_gt[gt["label"]]=[]
            box_gt[gt["label"]].append([float(i)*t_factor for i in gt["segment"]])
        gt_class_idx=[class_name_to_idx[cls_name] for cls_name in list(box_gt.keys())]

        # cas_gt=[s[:,gt_class_idx] for s in cas]
        act_gt=[a[:,gt_class_idx] for a in act_labels]
        bkg_gt=bkg_labels
        con_gt=[c[:,gt_class_idx] for c in con_labels]

        matchid_gt=[m[:,gt_class_idx] for m in matchids]
        weight_gt=[w[:,gt_class_idx] for w in weights]


        #load point anno
        temp_df = point_anno[point_anno["video_id"] == vid_name][['point', 'class']]
        point_gt=[]
        for key in temp_df['point'].keys():
            point = temp_df['point'][key]*t_factor/loc_gt['fps']
            point_gt.append(point)


        #plot
        # gs = gridspec.GridSpec(10, 1,wspace=0.3,hspace=0.3)
        # fig=plt.figure()
        # pseudo_ax=fig.add_subplot(gs[0:-1,:])
        # gt_ax=fig.add_subplot(gs[-1,:])

        # colors=['red','green','orange']
        # #plot pseudo_label
        # for j in range(len(steps)):
        #     #plot act seed
        #     for c in range(len(gt_class_idx)):
        #         color=colors[c]
        #         act_seed=act_gt[j][:,c]
        #         loc= np.where(act_seed>0)[0]
        #         psd_loc_label=np.split(loc, np.where(np.diff(loc) != 1)[0] + 1)

        #         for l,loc in enumerate(psd_loc_label):
        #             if len(loc)==0:
        #                 print('no act seed')
        #                 continue
        #             elif len(loc)==1:
        #                 s,e=loc[0],min(loc[0]+1,act_seed.shape[0])
        #             else:
        #                 s,e=loc[0],loc[-1]
        #             pseudo_ax.add_patch(plt.Rectangle(xy=(int(s),j+0.1),width=int(e)-int(s),height=0.8,
        #                             fill=True,color=color,ec=color,alpha=1,linestyle='-',linewidth=2))
        #     #plot bkg seed
        #     bkg_seed=bkg_gt[j]
        #     loc= np.where(bkg_seed>0)[0]
        #     bkg_label=np.split(loc, np.where(np.diff(loc) != 1)[0] + 1)
        #     for l,loc in enumerate(bkg_label):
        #         if len(loc)==0:
        #             continue
        #         elif len(loc)==1:
        #             s,e=loc[0],min(loc[0]+1,act_seed.shape[0])
        #         else:
        #             s,e=loc[0],loc[-1]
        #         pseudo_ax.add_patch(plt.Rectangle(xy=(int(s),j+0.1),width=int(e)-int(s),height=0.8,
        #                                 fill=True,color='gray',ec='gray',alpha=0.5,linestyle='-',linewidth=2))
        
        #     pseudo_ax.text(0,j+0.5,'step{}'.format(steps[j]),fontsize=10,ha="right",va="center")


        #     #plot context
        #     for c in range(len(gt_class_idx)):
        #         con_seed=con_gt[j][:,c]
        #         loc= np.where(con_seed>0)[0]
        #         con_label=np.split(loc, np.where(np.diff(loc) != 1)[0] + 1)

        #         for l,loc in enumerate(con_label):
        #             if len(loc)==0:
        #                 print('no con seed')
        #                 continue
        #             elif len(loc)==1:
        #                 s,e=loc[0],min(loc[0]+1,con_seed.shape[0])
        #             else:
        #                 s,e=loc[0],loc[-1]
        #             pseudo_ax.add_patch(plt.Rectangle(xy=(int(s),j+0.1),width=int(e)-int(s),height=0.8,
        #                             fill=True,color='purple',ec='purple',alpha=1,linestyle='-',linewidth=2))

        # #plot_gt
        # for cdx,class_name in enumerate(box_gt.keys()):
        #     color=colors[cdx]
        #     for loc in box_gt[class_name]:
        #         s,e=loc
        #         gt_ax.add_patch(plt.Rectangle(xy=(int(s),0),width=int(e)-int(s),
        #                         height=1,fill=True,color=color,alpha=1))
        # if len(point_gt)>0:
        #     for point in point_gt:
        #         gt_ax.axvline(x=int(point),color='black',linestyle='--',linewidth=1)

        # pseudo_ax.set_xticks([])
        # pseudo_ax.set_xlim([0,act_seed.shape[0]])
        # pseudo_ax.set_ylim([0,len(steps)])
        # gt_ax.set_xlim([0,act_seed.shape[0]])
        # fig.set_size_inches((20,15))
        # pseudo_ax.set_title(vid_name)
        # plt.savefig(os.path.join(save_path,'{}.png'.format(vid_name)),dpi=500)
        # plt.close()


        #plot_fig2
        colors=['red','green','blue','pink']
        num_step=len(steps)
        gs = gridspec.GridSpec(2*num_step+1, 1,wspace=0.3,hspace=0.3)
        fig=plt.figure()
        # pseudo_ax=fig.add_subplot(gs[0:-1,:])
        gt_ax=fig.add_subplot(gs[-1,:])

        #plot_gt
        gt_seq=np.zeros(T)
        for cdx,class_name in enumerate(box_gt.keys()):
            color=colors[cdx]
            for loc in box_gt[class_name]:
                s,e=loc
                gt_seq[int(s):int(e)+1]=1
                gt_ax.add_patch(plt.Rectangle(xy=(int(s),0),width=int(e)-int(s),
                                height=1,fill=True,color=color,alpha=1))   
        if len(point_gt)>0:
            for point in point_gt:
                gt_ax.axvline(x=int(point),color='black',linestyle='--',linewidth=1)
        

        for j in range(num_step):
            #plot weight
            w_ax=fig.add_subplot(gs[-2*(j+1),:])
            for c in range(len(gt_class_idx)):
                color=colors[c]
                weight_c=weight_gt[j][:,c]
                w_ax.plot(weight_c,color=color)
                w_ax.set_xticks([])
                w_ax.set_xlim([0,weight_c.shape[0]])
                w_ax.set_ylim([0,1])
            #plot match result
            m_ax=fig.add_subplot(gs[-2*j-3,:])
            for c in range(len(gt_class_idx)):
                matchid_c=matchid_gt[j][:,c]
                
                # matchid_c=matchid_c*gt_seq

                m_ax.plot(matchid_c,'o-',color=color)
                m_ax.set_xticks([])
                m_ax.set_yticks(range(5))
                m_ax.set_xlim([0,matchid_c.shape[0]])
                m_ax.set_ylim([0,4.1])


                act_seed=act_gt[j][:,c]
                loc= np.where(act_seed>0)[0]
                psd_loc_label=np.split(loc, np.where(np.diff(loc) != 1)[0] + 1)
                for l,loc in enumerate(psd_loc_label):
                    if len(loc)==0:
                        print('no act seed')
                        continue
                    elif len(loc)==1:
                        s,e=loc[0],min(loc[0]+1,act_seed.shape[0])
                    else:
                        s,e=loc[0],loc[-1]
                    m_ax.add_patch(plt.Rectangle(xy=(int(s),0),width=int(e)-int(s),height=4,
                                    fill=True,color=color,ec=color,alpha=0.5,linewidth=1))
                
                

        gt_ax.set_xlim([0,weight_c.shape[0]])
        fig.set_size_inches((20,15))
        # pseudo_ax.set_title(vid_name)
        plt.savefig(os.path.join(save_path,'{}_match.png'.format(vid_name)),dpi=500)
        plt.close()



if __name__ =="__main__":
    root_dir='/root/Weakly_TAL/baseline_v1/ckpt/THUMOS14/outputs/base_proto/proto(kmean_update_step50_maxmatch_cos_act0.3)'
    steps=np.arange(50,1000,200)
    # steps=[10,20,30]
    visual_pseudo_label(root_dir,steps,n=20)