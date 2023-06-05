import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset_extend import DE
from class_dict import class_dict
from utils_eval import segment_iou
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',family='Times New Roman') 
from pylab import mpl
mpl.rcParams['font.size'] = 12


class moa:
    def __init__(self,run_infos,dataset='thu',subset='train'):
        self.run_infos=run_infos
        self.subset=subset
        #plot info control
        self.plot_bkg=False
        self.plot_attn=True
        self.match_gt=False
        self.plot_video_scores=False
        self.plot_thresh=False
 
        self.class_dict=class_dict(dataset=dataset).class_dict
        #init path
        self.ckpt_path='./ckpt'

        if dataset=='thu':
            self.out_path=os.path.join(self.ckpt_path,'THUMOS14','outputs')
            self.act_thresh=0.5
        elif dataset=='act':
            self.out_path=os.path.join(self.ckpt_path,'ActivityNet13','outputs')
            self.act_thresh=0.5
        elif dataset=='beoid':
            self.out_path=os.path.join(self.ckpt_path,'BEOID','outputs')
            self.act_thresh=0.3
        elif dataset=='gtea':
            self.out_path=os.path.join(self.ckpt_path,'GTEA','outputs')
            self.act_thresh=0.5
        # self.score_thresh=0.4
        self.score_thresh=[0.5,0.5,0.4,0.6]
        self.cas_thresh=[[0,0.25],[0.5,0.725]]

        #init dataloader
        self.gt_loader=DE(dataset,subset)
        self.data_list=self.gt_loader.data_list#[0:1]#####################################################################
        self.gt_dict=self.gt_loader.gt_dict
        self.point_anno=self.gt_loader.point_anno

        #load prediction
        self.loc_preds_all={}
        for run_info in self.run_infos:
            temp_result_path=os.path.join(self.out_path,run_info,'temp_result_{}.json'.format(subset))
            assert os.path.exists(temp_result_path)
            with open(temp_result_path,'r') as json_file:
                loc_pred=json.load(json_file)
                self.loc_preds_all[run_info]=loc_pred


    def load_vid_data(self,vid_name):
        video_data_dict={}

        for run_info in self.run_infos:
            saved_file_path=os.path.join(self.out_path,run_info,'saved_file')
            video_data_dict[run_info]={}

            #load vid_score
            vid_score_path=os.path.join(saved_file_path,'vid_score',vid_name+'.csv')
            vid_score=pd.read_csv(vid_score_path)
            
            #get video-level prediction
            pred_classes=[]
            for index in range(len(vid_score)):
                if vid_score.values[index]>self.act_thresh:
                    pred_classes.append(self.class_dict[index])
            if len(pred_classes)==0:
                index=vid_score.iloc[:,0].argmax()
                pred_classes.append(self.class_dict[index])
            
            #load cas
            cas_path=os.path.join(saved_file_path,'cas_pred',vid_name+'.csv')
            cas_pred=pd.read_csv(cas_path)
            
            #load bkgscore
            if self.plot_bkg:
                bkg_path=os.path.join(saved_file_path,'bkg_score',vid_name+'.csv')
                bkg_score=pd.read_csv(bkg_path)
                video_data_dict[run_info]['bkg_score']=bkg_score

            #load attn
            # if os.path.exists(os.path.join(saved_file_path,'self_attn')):
            #     self_attn=pd.read_csv(os.path.join(saved_file_path,'self_attn',vid_name+'.csv'))
            #     video_data_dict[run_info]['self_attn']=self_attn
            #     cross_attn=pd.read_csv(os.path.join(saved_file_path,'cross_attn',vid_name+'.csv'))
            #     video_data_dict[run_info]['cross_attn']=cross_attn

            #load groud truth
            loc_gt=self.gt_dict[vid_name]
            self.t_factor=cas_pred.shape[0]/loc_gt['duration']

            if self.subset=='train':#load point label if avialable
                temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
                point_gt=[]
                for key in temp_df['point'].keys():
                    point = temp_df['point'][key]*self.t_factor/loc_gt['fps']
                    point_gt.append(point)
                video_data_dict[run_info]['point_gt']=point_gt
            
            box_gt={}#sort by class
            for gt in loc_gt['annotations']:
                if gt["label"] not in box_gt.keys():
                    box_gt[gt["label"]]=[]
                box_gt[gt["label"]].append([float(i)*self.t_factor for i in gt["segment"]])
            
            #load prediction
            loc_preds=self.loc_preds_all[run_info]['results'][vid_name]
            box_pred={}#sort by class
            for pred in loc_preds:
                if pred["label"] not in box_pred.keys():
                    box_pred[pred["label"]]=[]
                box_pred[pred["label"]].append([i*self.t_factor for i in pred["segment"]]+[pred["score"]])  #[start ,end ,score]
            if self.match_gt:#reduce redundant predictions
                box_pred=self.match_gtbox(box_pred,box_gt)
            
            #save to dict
            video_data_dict[run_info]['vid_score']=vid_score
            video_data_dict[run_info]['pred_classes']=pred_classes
            video_data_dict[run_info]['cas_pred']=cas_pred
            video_data_dict[run_info]['box_gt']=box_gt
            video_data_dict[run_info]['box_pred']=box_pred 

        return video_data_dict

    def match_gtbox(self,preds,gt):
        pred_box={}
        for cla,pred in preds.items():
            if cla not in gt.keys():
                continue
                
            pred_box[cla]=[]
            lock_gt=np.ones(len(gt[cla]))
            for seg in pred:
                iou=segment_iou(np.array(seg[0:2]),np.array(gt[cla]))
                match_gt=np.argmax(iou)
                if lock_gt[match_gt]>0:
                    pred_box[cla].append(seg)
                    lock_gt[match_gt]=0
        return pred_box

    def plot_vs(self,vs_ax1,vs_ax2,video_data_dict):
        vs_axs=[vs_ax1,vs_ax2]
        for vs_ax,run_info in zip(vs_axs,self.run_infos):
            vid_score=video_data_dict[run_info]['vid_score']#vid_score[numclass,1]
            pred_classes=video_data_dict[run_info]['pred_classes']
            for cid in range(vid_score.shape[0]): 
                color="green" if self.class_dict[cid] in pred_classes else "gray"
                score=vid_score.iloc[:,0].values[cid]
                b=vs_ax.barh(cid,score,color=color)
                for rect in b:
                    w=rect.get_width()
                    tag="{}:{:.2f}".format(list(self.class_dict.values())[cid],w)
                    vs_ax.text(w/4,rect.get_y()+rect.get_height()/2,tag,fontsize=10,ha="left",va="center")
            vs_ax.set_xticks([])
            vs_ax.set_yticks([])
            for line in ["right","top","bottom"]:
                vs_ax.spines[line].set_visible(False)



    def visualize_data(self,video_data_dict,vid_name,file_path):
        num_run=len(self.run_infos)
        cas_ax_cols=8
        cas_ax_row=1+2*num_run +2 if self.plot_attn else 1+2*num_run 
        nums_axcol=cas_ax_cols+2 if self.plot_video_scores else cas_ax_cols
        nums_axrow=cas_ax_row+3 if self.plot_bkg else cas_ax_row
        gs = gridspec.GridSpec(nums_axrow, nums_axcol,wspace=0.3,hspace=0.3)
        fig=plt.figure()
        gt_ax=fig.add_subplot(gs[0,0:cas_ax_cols])

        
        # if self.plot_bkg:
        #     bkg_ax=fig.add_subplot(gs[3:6,0:cas_ax_cols])
        
        
        # if self.plot_video_scores:
        #     vs_ax1=fig.add_subplot(gs[:,cas_ax_cols:cas_ax_cols+1])
        #     vs_ax2=fig.add_subplot(gs[:,cas_ax_cols+1:cas_ax_cols+2])
        #     self.plot_vs(vs_ax1,vs_ax2,video_data_dict)
        
        colors=['red','blue','orange','green','purple','pink','magenta']
        # colors=['#99ff99','blue','#ff3333']
        linestyles=['-','--','-.',':']
        lw=3
        class_color_dict={}
        for i,run_info in enumerate(self.run_infos):
            cas_ax=fig.add_subplot(gs[2*i+1,0:cas_ax_cols])
            pred_ax=fig.add_subplot(gs[2*i+2,0:cas_ax_cols])

            #plot background score
            # if self.plot_bkg:
            #     bkg_score=video_data_dict[run_info]['bkg_score']
            #     bkg_ax.plot(1-bkg_score,label="{}-Actioness({})".format(i,run_info),color=colors[i%len(colors)],linestyle='--',linewidth=lw)

            #plot cas and prediction
            cas_pred=video_data_dict[run_info]['cas_pred']
            pred_classes=video_data_dict[run_info]['pred_classes']
            box_pred=video_data_dict[run_info]['box_pred']
            for cid in range(cas_pred.shape[1]):
                linestyle='-'
                # color=colors[(cid+i)%len(colors)]
                color='blue'

                class_name=pred_classes[cid]
                class_color_dict[class_name]=color
                #plot cas curve
                cas_ax.plot(cas_pred.iloc[:,cid],label="{}-{}({})".format(i,class_name,run_info),color=color,linestyle=linestyle,linewidth=lw)
                #plot loc pred
                if class_name not in box_pred.keys():
                    continue
                for loc in box_pred[class_name]:
                    s,e,score=loc
                    if score>self.score_thresh[i]:
                        if score>0.63 and score<0.65:
                            continue
                        pred_ax.add_patch(plt.Rectangle(xy=(int(s),0),width=int(e)-int(s),height=1,
                                fill=True,color=color,ec=color,alpha=1,linestyle=linestyle,linewidth=2))#min(1,score**3)
                        # if self.match_gt:
                        # pred_ax.text(int(s)+0.5*(int(e)-int(s)),0.5,str(round(score,2)),color='r',fontsize=20,ha="center",va="center")

            cas_ax.set_xticks([])
            cas_ax.set_yticks([])
            cas_ax.set_xlim([0,cas_pred.shape[0]])
            cas_ax.set_ylim([0,1.05])
            
            pred_ax.set_xticks([])
            pred_ax.set_yticks([])
            pred_ax.set_xlim([0,cas_pred.shape[0]])
            pred_ax.set_ylim([0,1])

        #plot groud truth
        box_gt=video_data_dict[run_info]['box_gt']
        for class_name in box_gt.keys():
            color=class_color_dict[class_name] if class_name in box_pred.keys() else "r"
            for loc in box_gt[class_name]:
                s,e=loc
                gt_ax.add_patch(plt.Rectangle(xy=(int(s),0),width=int(e)-int(s),
                                    height=1,fill=True,color='orange',alpha=1))

        #plot point label if availabel
        if self.subset=='train':
            point_gt=video_data_dict[run_info]['point_gt']
            if len(point_gt)>0:
                for point in point_gt:
                    gt_ax.axvline(x=int(point),color='black',linestyle='--',linewidth=1)

        #plot cas_thresh
        if self.plot_thresh:
            [act_low,act_up],[agno_low,agno_up]=self.cas_thresh
            cas_ax.add_patch(plt.Rectangle(xy=(0,act_low),width=cas_pred.shape[0],
                                    height=act_up-act_low,fill=True,color='red',alpha=0.2))
            # if self.plot_bkg:
            #     bkg_ax.add_patch(plt.Rectangle(xy=(0,agno_low),width=cas_pred.shape[0],
            #                             height=agno_up-agno_low,fill=True,color='gray',alpha=0.2))
            #     bkg_ax.legend(bbox_to_anchor=(-0.02,1),loc='upper right',frameon=False,fontsize=10)
            #     bkg_ax.set_xticks([])
            #     bkg_ax.set_xlim([0,cas_pred.shape[0]])
            #     bkg_ax.set_ylim([0,1.1])
                    
        # cas_ax.legend(bbox_to_anchor=(-0.02,1),loc='upper right',frameon=False,fontsize=10)
        gt_ax.set_xticks([])
        gt_ax.set_yticks([])
        gt_ax.set_xlim([0,cas_pred.shape[0]])

        
        fig.set_size_inches((25,10))
        # cas_ax.set_title(vid_name)
        plt.savefig(file_path,dpi=400)
        plt.close()

    def run(self,info):
        save_path='./tools/figs/model_outputs_{}_{}'\
                .format('match' if self.match_gt else 'all',info)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for idx,vid_name in tqdm(enumerate(self.data_list),total=len(self.data_list)):
            file_path=os.path.join(save_path,'{}.png'.format(vid_name))
            # if not os.path.exists(file_path):
            video_data_dict=self.load_vid_data(vid_name)
            self.visualize_data(video_data_dict,vid_name,file_path)
            


if __name__ == '__main__':
    import random
    random.seed(0)
    # run_infos=['selfattn_countloss/0.2_2layer_0.3drop','selfattn_countloss/0.2_2layer_0.3drop_initargs']
    # run_infos=['baseline/base','selfattn_countloss/0.2_2layer_0.3drop']
    # run_infos=['SC/base_300','SC/base_best','SC/base_500']

    # info='QR_base_CoRL'
    # run_infos=['qualitative_result/base','qualitative_result/CoRL']#,'base_proto/proto','base_proto/proto(+CA)','point_detector/proto_mask(SAmask_CA)(video0_pN5)']
    
    # info='Attn_base_CoRL'
    # run_infos=['qualitative_result/baseline','qualitative_result/CoRL']

    info='abl_base_CoRL_good'
    run_infos=['qualitative_result/baseline','qualitative_result/baseline+CVR','qualitative_result/baseline+IVR','qualitative_result/CoRL']
    dataset='thu'

    # Analyser=moa(run_infos,dataset=dataset,subset='train')
    # select_idx=np.random.choice(len(Analyser.data_list),20)
    # Analyser.data_list=np.array(Analyser.data_list)[select_idx]
    # Analyser.run(info)

    Analyser=moa(run_infos,dataset=dataset,subset='test')
    # select_idx=np.sort(random.sample(range(len(Analyser.data_list)),50))
    # Analyser.data_list=np.array(Analyser.data_list)[select_idx]
    Analyser.data_list=[item.strip() for item in list(open('./good_case_test.txt'))]
    Analyser.run(info)



#测试anet上的效果，并调整
#增加关键帧显示