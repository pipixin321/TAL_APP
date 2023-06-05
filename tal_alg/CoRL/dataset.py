import os 
import json 
import torch 
import numpy as np 
import pandas as pd
from torch.utils.data import Dataset 
import random
import pickle


class dataset(Dataset):
    def __init__(self,args,phase="train",sample="random",stage=2,GT=False):
        self.phase=phase
        self.sample=sample
        self.data_path=args.data_path
        self.num_segments = args.num_segments
        self.frames_per_sec=args.frames_per_sec
        self.supervision = args.supervision
        self.stage=stage
        self.args=args
        self.GT=GT

        with open(os.path.join(self.data_path,"gt_full.json")) as gt_f:
            self.gt_dict=json.load(gt_f)["database"]
    
        if self.phase == "train":
            self.feature_dir = os.path.join(self.data_path, "train")
            self.data_list = [item.strip() for item in list(open(os.path.join(self.data_path, "split_train.txt")))]
            print("number of train videos:{}".format(len(self.data_list)))
        else:
            self.feature_dir = os.path.join(self.data_path, "test")
            self.data_list = [item.strip() for item in list(open(os.path.join(self.data_path, "split_test.txt")))]
            print("number of test videos:{}".format(len(self.data_list)))
        self.feature_dir="/mnt/data1/zhx/TAL_APP/datasets/THUMOS14/features/swin_tiny"
        self.class_name_lst = args.class_name_lst
        self.action_class_idx_dict={action_cls:idx for idx, action_cls in enumerate(self.class_name_lst)}
        self.action_class_num = args.action_cls_num

        self.label_dict = {}
        self.count_dict={}
        for item_name in self.data_list:
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = np.zeros(self.action_class_num)
            count_label=np.zeros(self.action_class_num)
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label[self.action_class_idx_dict[ann_label]] = 1.0
                count_label[self.action_class_idx_dict[ann_label]]+=1.0
            self.label_dict[item_name] = item_label
            self.count_dict[item_name]=count_label

        if self.supervision == 'point':
            if self.args.manual_id!=0:
                self.point_annos =[]
                for manual_id in range(1,5):
                    point_anno = pd.read_csv(os.path.join(self.data_path, 'point_labels', 'point_labels_manual{}.csv'.format(manual_id)))
                    self.point_annos.append(point_anno)
                self.manual_id_list=[np.random.choice(range(len(self.point_annos))) for i in range(len(self.data_list))]
            else:
                self.point_anno = pd.read_csv(os.path.join(self.data_path, 'point_gaussian', 'point_labels.csv'))

        self.pseudo_label={'act_seq':[-1]*len(self.data_list),'bkg_seq':[-1]*len(self.data_list),
                            'act_attn_mask':[-1]*len(self.data_list),'bkg_attn_mask':[-1]*len(self.data_list)}
        
        #class datalist
        self.class_data_list={c:[] for c in range(self.action_class_num )}
        for item_name in self.data_list:
            item_vid_label=self.label_dict[item_name]
            vid_class=np.nonzero(item_vid_label)[0]
            for vc in vid_class:
                self.class_data_list[vc].append(item_name)

    
    def get_gt(self, idx, vid_len):
        vid_name=self.data_list[idx]
        vid_gt = self.gt_dict[vid_name]
        temp_anno = np.zeros([vid_len, self.action_class_num], dtype=np.float32)
        # t_factor = self.frames_per_sec / (vid_gt["fps"] * 16)
        t_factor = vid_len / (vid_gt["fps"] * vid_duration) 
        for anno in vid_gt['annotations']:
            s,e=anno['segment(frames)']
            class_idx=self.action_class_idx_dict[anno['label']]
            temp_anno[int(s * t_factor):int(e * t_factor),class_idx] = 1
        return torch.from_numpy(temp_anno)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        vid_name=self.data_list[idx]
        data, vid_len, sample_idx=self.get_data(vid_name)
        vid_label,point_anno,vid_duration=self.get_label(idx,vid_len,sample_idx)
        gt_heatmap=self.get_heatmap(point_anno)

        count=self.count_dict[vid_name]
        count=torch.as_tensor(count.astype(np.float32))

        pseudo_label={'act_seq':self.pseudo_label['act_seq'][idx],'bkg_seq':self.pseudo_label['bkg_seq'][idx],
                        'act_attn_mask':self.pseudo_label['act_attn_mask'][idx],
                        'bkg_attn_mask':self.pseudo_label['bkg_attn_mask'][idx]}

        mask_seq=[-1]*vid_len
        # if self.stage==2:
        #     mask_seq=np.load(os.path.join(self.args.data_path,'mask',vid_name+".npy"))
        #     mask_seq=torch.as_tensor(mask_seq.astype(np.float32))
        if self.GT:
            point_anno=self.get_gt(idx,vid_len)
            
        return idx, data, vid_label, point_anno, mask_seq, gt_heatmap, self.data_list[idx], vid_len, vid_duration

    def get_heatmap(self,point_anno):
        gt_heatmap=point_anno.max(dim=1,keepdim=True)[0]
        return gt_heatmap
    

    def get_data(self,vid_name):
        # con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name+".npy"))
        f=open(os.path.join(self.feature_dir, vid_name+".pkl"), 'rb')
        con_vid_feature = pickle.load(f)
        f.close()

        vid_len = con_vid_feature.shape[0]

        if self.sample == "random":
            sample_idx = self.random_perturb(vid_len)
        elif self.sample == 'uniform':
            sample_idx = self.uniform_sampling(vid_len)
        else:
            raise AssertionError('Not supported sampling !')

        feature = con_vid_feature[sample_idx]

        return torch.as_tensor(feature.astype(np.float32)), vid_len, sample_idx

    def get_label(self, idx, vid_len, sample_idx):
        vid_name=self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_gt = self.gt_dict[vid_name]
        vid_duration=vid_gt['duration']
        #Following SF-Net,we random select munual id of point labels while training
        if self.args.manual_id!=0:
            self.point_anno=self.point_annos[self.manual_id_list[idx]]

        if self.supervision == 'video':
            return torch.as_tensor(vid_label.astype(np.float32)), torch.Tensor(0),vid_duration

        elif self.supervision == 'point' and self.num_segments==-1:
            temp_anno = np.zeros([vid_len, self.action_class_num], dtype=np.float32)
            # t_factor = self.frames_per_sec / (vid_gt["fps"] * 16) if self.args.dataset=='THUMOS14' else vid_len / (vid_gt["fps"] * vid_duration) 
            t_factor = vid_len / (vid_gt["fps"] * vid_duration) 
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = self.action_class_idx_dict[temp_df['class'][key]]
                temp_anno[int(point * t_factor)][class_idx] = 1
            point_label = temp_anno[sample_idx, :]
            self.check_pointlabels(point_label)

            return torch.as_tensor(vid_label.astype(np.float32)), torch.from_numpy(point_label),vid_duration
        
        #avoid absence of point label
        elif self.supervision == 'point' and self.num_segments>0:
            temp_anno = np.zeros([self.num_segments, self.action_class_num], dtype=np.float32)
            t_factor = self.num_segments / (vid_gt["fps"] * vid_duration)
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class']]
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = self.action_class_idx_dict[temp_df['class'][key]]
                temp_anno[int(point * t_factor)][class_idx] = 1
            point_label = temp_anno
            # self.check_pointlabels(point_label)

            return torch.as_tensor(vid_label.astype(np.float32)), torch.from_numpy(point_label),vid_duration

    def check_pointlabels(self,point_label):
        point_label=torch.from_numpy(point_label)
        point_anno_agnostic=point_label.max(dim=1)[0]
        act_idx = torch.nonzero(point_anno_agnostic).squeeze(1)
        if self.phase=='train' and len(act_idx)==0:
            print("no point label sampled")

    def random_perturb(self, length):
        if self.num_segments == length or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if length <= self.num_segments or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)


    def split_data(self,data,point_anno,act_idx,method=1):
        num_act=act_idx.shape[0]
        data_split,point_anno_split=[],[]

        if method==1:
            data_split.append(data[:act_idx[0]-1])
            point_anno_split.append(point_anno[:act_idx[0]-1])
            for i in range(num_act-1):
                if act_idx[i+1]-act_idx[i]<=1:
                    continue
                data_split.append(data[act_idx[i]:act_idx[i+1]-1])
                point_anno_split.append(point_anno[act_idx[i]:act_idx[i+1]-1])
            data_split.append(data[act_idx[-1]:])
            point_anno_split.append(point_anno[act_idx[-1]:])

        elif method==2:
            for i in range(num_act):
                s=0 if i==0 else act_idx[i-1]
                e=data.shape[0]-1 if i==num_act-1 else act_idx[i+1]
                data_split.append(data[s:e])
                point_anno_split.append(point_anno[s:e])


        return data_split,point_anno_split


    def random_combo(self,idx,data,vid_label,point_anno):
        r_ratio=0.1

        vid_name=self.data_list[idx]
        choice_lst=[]
        gt_glass=np.nonzero(vid_label.numpy())[0]
        for c in gt_glass:
            choice_lst+=self.class_data_list[c]
        choice_lst.remove(vid_name)

        #random choose replace video
        # choice_lst=list(range(len(self.data_list)))
        # choice_lst.remove(idx)
        # r_idx=random.choice(choice_lst)
        # r_vid_name=self.data_list[r_idx]

        r_vid_name=random.choice(choice_lst)
        r_idx=self.data_list.index(r_vid_name)
        r_data, r_vid_len, r_sample_idx=self.get_data(r_vid_name)
        r_vid_label,r_point_anno,r_vid_duration=self.get_label(r_idx,r_vid_len,r_sample_idx)
        

        #replace
        point_anno_agnostic=point_anno.max(dim=1)[0]
        act_idx=torch.nonzero(point_anno_agnostic).squeeze(1)
        data_split,point_anno_split=self.split_data(data,point_anno,act_idx,method=1)
        

        r_point_anno_agnostic=r_point_anno.max(dim=1)[0]
        r_act_idx=torch.nonzero(r_point_anno_agnostic).squeeze(1)
        r_data_split,r_point_anno_split=self.split_data(r_data,r_point_anno,r_act_idx,method=2)
        
        num_act=len(data_split)
        r_num_act=len(r_data_split)
        r_num=min(int((num_act)*r_ratio),r_num_act)
        idx_orig=np.sort(random.sample(list(range(num_act)),num_act-r_num))
        idx_rep=np.sort(random.sample(range(r_num_act),r_num))

        new_data,new_point_anno=[],[]
        j=0
        for i in range(num_act):
            if i in idx_orig:
                new_data.append(data_split[i])
                new_point_anno.append(point_anno_split[i])
            else:
                new_data.append(r_data_split[idx_rep[j]])
                new_point_anno.append(r_point_anno_split[idx_rep[j]])
                j+=1

        new_data=torch.cat(new_data)
        new_point_anno=torch.cat(new_point_anno)
        new_vid_label=new_point_anno.max(dim=0)[0]
        
        # print('replace:{}/{}'.format(r_num,num_act))
        return new_data,new_vid_label,new_point_anno

if __name__ == '__main__':
    import options
    args=options.parse_args()
    data=dataset(args)
    
    np.random.seed(0)
    random.seed(0)
    for i in range(200):
        idx, b, vid_label, point_anno, mask_seq, gt_heatmap, a, vid_len, vid_duration=data.__getitem__(i)
    
 
