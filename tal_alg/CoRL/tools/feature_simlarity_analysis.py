import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import spatial
from tqdm import tqdm
from  dataset_extend import DE

def plot_cosine_similarity(data,vid_label,point_label,instance_label,vid_name,dataset):
    save_path='/mnt/data1/zhx/Weakly_TAL/baseline/tools/figs/cos_similarity_{}'.format(dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path=os.path.join(save_path,'{}.png'.format(vid_name))
    
    point_idx=np.nonzero(point_label)[0]
    num_point=len(point_idx)
    if num_point==len(instance_label) and not os.path.exists(file_path):
        fig=plt.figure()
        point_cosine_similarity=np.zeros([num_point,data.shape[0]])
        for idx in range(len(point_idx)):
            point_feat=data[point_idx[idx],:]
            for jdx in range(data.shape[0]):
                cos_sim = 1 - spatial.distance.cosine(point_feat, data[jdx,:])
                point_cosine_similarity[idx,jdx]=cos_sim
        # print(instance_label)

        colors = ['red','green','orange','blue','pink','magenta','purple']
        for idx in range(num_point):
            color=colors[idx%len(colors)]
            plt.axvline(x=point_idx[idx],color=color,linestyle="--",linewidth=1)
            plt.plot(point_cosine_similarity[idx,:],color=color,label=vid_label[idx])

            instance=instance_label[idx]
            x=np.arange(instance[0],instance[1])
            y=1.1*np.ones([x.shape[0]])
            plt.plot(x,y,color=color)
            
        plt.title(vid_name)
        fig.set_size_inches((10,2))
        plt.savefig(file_path)
        plt.close()


if __name__=="__main__":
    dataset='act'
    database=DE(dataset=dataset,subset="train")
    for i in tqdm(range(len(database))):
        data,vid_label,point_label,instance_label,vid_name=database.__getitem__(i)
        plot_cosine_similarity(data,vid_label,point_label,instance_label,vid_name,dataset)

    dataset='thu'
    database=DE(dataset=dataset,subset="train")
    for i in tqdm(range(len(database))):
        data,vid_label,point_label,instance_label,vid_name=database.__getitem__(i)
        plot_cosine_similarity(data,vid_label,point_label,instance_label,vid_name,dataset)
    


