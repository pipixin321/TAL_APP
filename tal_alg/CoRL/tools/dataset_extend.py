from torch.utils.data import Dataset 
import pandas as pd
import json
import numpy as np
import os

class DE(Dataset):
    def __init__(self,dataset='thu',subset="train"):
        if dataset=='thu':
            self.data_path="./dataset/THUMOS14/"
            self.num_segments=750
        elif dataset=='act':
            self.data_path="./dataset/ActivityNet13"
            self.num_segments=75
        elif dataset=='beoid':
            self.data_path="./dataset/BEOID"
            self.num_segments=750
        elif dataset=='gtea':
            self.data_path="./dataset/GTEA"
            self.num_segments=1000
        self.subset=subset
        
        self.feature_dir = os.path.join(self.data_path, subset)
        self.data_list = [item.strip() for item in list(open(os.path.join(self.data_path, "split_{}.txt".format(subset))))]
        print("number of videos:{}".format(len(self.data_list)))

        with open(os.path.join(self.data_path,"gt_full.json")) as gt_f:
            self.gt_dict=json.load(gt_f)["database"]
        self.label_dict = {}
        for item_name in self.data_list:
            item_anns_list = self.gt_dict[item_name]["annotations"]
            item_label = []
            for ann in item_anns_list:
                ann_label = ann["label"]
                item_label.append(ann_label)
            self.label_dict[item_name] = item_label
        self.point_anno = pd.read_csv(os.path.join(self.data_path, 'point_gaussian', 'point_labels.csv'))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data=self.get_data(idx)
        vid_label,point_label,instance_label,vid_name=self.get_label(idx)

        return data,vid_label,point_label,instance_label,vid_name

    def get_data(self,idx):
        vid_name=self.data_list[idx]
        con_vid_feature = np.load(os.path.join(self.feature_dir, vid_name+".npy"))
        vid_len = con_vid_feature.shape[0]
        sample_idx = self.random_perturb(vid_len)
        feature = con_vid_feature[sample_idx]

        return feature.astype(np.float32)

    def get_label(self, idx):
        vid_name=self.data_list[idx]
        vid_label = self.label_dict[vid_name]
        vid_gt = self.gt_dict[vid_name]
        vid_duration=vid_gt['duration']

        point_level_label = np.zeros(self.num_segments, dtype=np.float32)
        instance_level_label=[]
        t_factor = self.num_segments / (vid_gt["fps"] * vid_duration)  
        temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['start_frame','stop_frame','point', 'class']]
        for key in temp_df['point'].keys():
            point = temp_df['point'][key]
            start=temp_df['start_frame'][key]
            stop=temp_df['stop_frame'][key]
            point_level_label[int(point * t_factor)]= 1
            instance_level_label.append([int(start*t_factor),int(stop*t_factor)+1])

        return vid_label,point_level_label.astype(np.float32),instance_level_label,vid_name

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