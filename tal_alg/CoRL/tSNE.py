import torch
import torch.utils.data as data
from dataset import dataset
import options
from model import Model

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rc('font',family='Times New Roman') 
from pylab import mpl
mpl.rcParams['font.size'] = 12

def cal_featlist(feats,act_seq,vid_label):
    featlist={}
    for b in range(act_seq.shape[0]):
        gt_class=torch.nonzero(vid_label[b]).cpu().squeeze(1).numpy()

        for c in gt_class:
            act_id=torch.nonzero(act_seq[b,:,c]).squeeze(1)

            act_feat=feats[b,act_id,:]
            if c not in featlist.keys():
                featlist[c]=act_feat
            else:
                featlist[c]=torch.cat(featlist[c],act_feat)

    return featlist #featlist{c_id:tensor[T,D]}


def main(args,load_cache,info,plot_proto):
    #init artgs
    t_start =time.perf_counter() 
    
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = device

    #dataloader
    train_loader=data.DataLoader(dataset(args,phase="train",sample="random",stage=args.stage,GT=True),
                                                batch_size=1,shuffle=True, num_workers=args.num_workers)
    test_loader=data.DataLoader(dataset(args,phase="test",sample="random",stage=args.stage,GT=True),
                                                batch_size=1,shuffle=False, num_workers=args.num_workers)

    #load model
    net=Model(args)
    checkpoint = torch.load(os.path.join(args.model_path,"model_seed_{}.pkl".format(args.seed)), map_location=torch.device("cpu"))
    net.load_state_dict(checkpoint)
    net=net.to(device)


    #inference
    feat_total={}
    classes=args.class_name_lst
    c_lst=list(range(20))


    loader = train_loader
    with torch.no_grad():
        net.eval()
        load_iter = iter(loader)
        for i in tqdm(range(len(loader.dataset))):
            _idx, _data, vid_label, _point_anno, mask_seq, gt_heatmap, vid_name, vid_len, vid_duration=next(load_iter)
            _data=_data.to(args.device)
            vid_label=vid_label.to(args.device)
            vid_score,embeded_feature,cas_sigmoid_fuse,_,_,_=net(_data,None,None,mask_seq)

            embeded_feature=embeded_feature.detach().cpu()
            intial_feat_list=cal_featlist(embeded_feature,_point_anno,vid_label)
            for c in intial_feat_list.keys():
                if c not in feat_total.keys():
                    feat_total[c]=intial_feat_list[c]
                else:
                    feat_total[c]=torch.cat([feat_total[c],intial_feat_list[c]],dim=0)

    # loader = test_loader
    # with torch.no_grad():
    #     net.eval()
    #     load_iter = iter(loader)
    #     for i in tqdm(range(len(loader.dataset))):
    #         _idx, _data, vid_label, _point_anno, mask_seq, gt_heatmap, vid_name, vid_len, vid_duration=next(load_iter)
    #         _data=_data.to(args.device)
    #         vid_label=vid_label.to(args.device)
    #         vid_score,embeded_feature,cas_sigmoid_fuse,_,_,_=net(_data,None,None,mask_seq)

    #         embeded_feature=embeded_feature.detach().cpu()
    #         intial_feat_list=cal_featlist(embeded_feature,_point_anno,vid_label)
    #         for c in intial_feat_list.keys():
    #             if c not in feat_total.keys():
    #                 feat_total[c]=intial_feat_list[c]
    #             else:
    #                 feat_total[c]=torch.cat([feat_total[c],intial_feat_list[c]],dim=0)


    #preprocess data
    X=torch.cat([feat_total[c] for c in c_lst],dim=0).numpy()
    Y=[]
    for c in c_lst:
        c_label=[int(c)]*feat_total[c].shape[0]
        Y+=c_label
    Y=np.array(Y).reshape(-1,1)

    #preprocess prototype
    if plot_proto:
        proto=net.state_dict()['cross_attn.0.proto'].detach().cpu()
        X_proto=torch.cat([proto[c] for c in c_lst],dim=0).numpy()
        Y_proto=[]
        for c in c_lst:
            c_label=[int(c)]*proto.shape[1]
            Y_proto+=c_label
        Y_proto=np.array(Y_proto).reshape(-1,1)

        X=np.vstack([X,X_proto])
        Y=np.vstack([Y,Y_proto])


    #T-SNE process
    save_path='/root/Weakly_TAL/baseline_v1/tools/figs/tSNE'
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    import seaborn as sns
    if load_cache:
        X_tsne=np.load(os.path.join(save_path,'X_tsne.npy'),allow_pickle=True)
    else:
        pca=PCA(n_components=50,random_state=0)
        X=pca.fit_transform(X)
        tsne = TSNE(n_components=2,random_state=0,verbose=1)
        X_tsne=tsne.fit_transform(X)
        np.save(os.path.join(save_path,'X_tsne.npy'),X_tsne)
    print(X.shape,X_tsne.shape,Y.shape)
    print("TSNE Done!")

    #plot result
    X_tsne_data = np.hstack((X_tsne, Y))
    df_tsne = pd.DataFrame(X_tsne_data, columns=['x','y','class'])
    plt.figure(figsize=(4, 4))
    if plot_proto:
        df_base=df_tsne.iloc[:-proto.shape[0]*proto.shape[1],:]
        sns.scatterplot(data=df_base, hue='class', x='x', y='y',palette='tab20', legend=None, linewidth=0.1, s=3, alpha=0.5)

        df_proto=df_tsne.iloc[-proto.shape[0]*proto.shape[1]:,:]
        sns.scatterplot(data=df_proto, hue='class', x='x', y='y',palette='tab20', legend=None, linewidth=0.2, s=50, alpha=1, marker='*',edgecolor='black')
    else:
        sns.scatterplot(data=df_tsne, hue='class', x='x', y='y',palette='tab20', legend=None, linewidth=0.1, s=3, alpha=0.5)
    # plt.legend(bbox_to_anchor=(-0.02,1),loc='upper right',frameon=False,fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    #save result
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path,'{}.png'.format(info)),dpi=500)

    t_end = time.perf_counter()
    print('Running time: %s Seconds'%(t_end-t_start))



if __name__=="__main__":
    args=options.parse_args()
    info=args.run_info.split('/')[-1]+'train_set'
    load_cache=False
    plot_proto=False
    main(args,load_cache,info,plot_proto)
