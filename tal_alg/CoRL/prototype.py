import torch
import torch.utils.data as data
import torch.nn.functional as F
from kmeans_pytorch import kmeans
import utils

class Prototypes():
    def __init__(self,train_loader,class_number,feat_dim):
        self.train_loader=train_loader

        self.class_number=class_number
        self.feat_dim=feat_dim
        self.proto_num=5
        self.proto_momentum=0.001 #0.0001
        self.proto_temperature=1
        self.proto_vectors=torch.zeros([self.class_number,self.proto_num,self.feat_dim])
        self.pro_vectors_num=torch.zeros([self.class_number])
        # self.init_proto_from_point()

    def init_proto_from_point(self,args,net):
        #inital prototype from point feature
        feat_total={}
        gt_point_num=torch.zeros([self.class_number])
        temp_loader = data.DataLoader(self.train_loader.dataset, batch_size=1,shuffle=True, num_workers=4)
        loader_iter = iter(temp_loader)
        for i in range(len(self.train_loader.dataset)):
            _, _data, vid_label, point_anno, mask_seq, _, _, _, _ = next(loader_iter)
            with torch.no_grad():
                net.eval()
                _,embeded_feature,_,_,_,_=net(_data.to(args.device), self.proto_vectors.to(args.device), vid_label.to(args.device))

            # point_anno=self.align_mask(point_anno,mask_seq)
            intial_feat_list,num_points=self.cal_featlist(embeded_feature,point_anno,vid_label)
            for c in intial_feat_list.keys():
                if c not in feat_total.keys():
                    feat_total[c]=intial_feat_list[c]
                else:
                    feat_total[c]=torch.cat([feat_total[c],intial_feat_list[c]],dim=0)

        #kmeans for each class
        for c in range(self.class_number):
            if self.proto_num>1:
                center_id, cluster_centers = kmeans(X=feat_total[c], num_clusters=self.proto_num, distance='euclidean', device=args.device)
            else:
                cluster_centers=feat_total[c].mean(dim=0,keepdim=True)
            self.proto_vectors[c]=cluster_centers

    def align_mask(self,point_anno,mask_seq):
        align_mask=torch.zeros(point_anno.shape)
        for b in range(point_anno.shape[0]):
            mask_seq=mask_seq[b]
            seq_diff=mask_seq[1:]-mask_seq[:-1]
            range_idx=torch.nonzero(seq_diff).squeeze(1)
            range_idx=range_idx.cpu().data.numpy().tolist()
            if type(range_idx) is not list:
                range_idx = [range_idx]
            if len(range_idx) == 0:
                return
            if seq_diff[range_idx[0]] != 1:
                range_idx = [-1] + range_idx 
            if seq_diff[range_idx[-1]] != -1:
                range_idx = range_idx + [seq_diff.shape[0] - 1]

            for i in range(len(range_idx)//2):
                s,e=range_idx[2*i],range_idx[2*i+1]
                if e-s<1:
                    continue
                for c in range(point_anno.shape[-1]):
                    c_anno=point_anno[b,:,c]
                    point_idx=torch.nonzero(c_anno).squeeze(1)
                    if len(point_idx)>0:
                        for p_idx in point_idx:
                            if p_idx>=s and p_idx<=e:
                                align_mask[b,s:e+1,c]=1
                                break

        return align_mask

    def update_prototype_kmean(self,args,net,step):
        #inital prototype from point feature
        feat_total={}

        temp_loader = data.DataLoader(self.train_loader.dataset, batch_size=1, shuffle=True, num_workers=4)
        loader_iter = iter(temp_loader)
        for i in range(len(self.train_loader.dataset)):
            _, _data, vid_label, point_anno, mask_seq, _, _, _, _ = next(loader_iter)
            with torch.no_grad():
                net.eval()
                _,embeded_feature,cas_sigmoid_fuse,_,_,_=net(_data.to(args.device), self.proto_vectors.to(args.device), vid_label.to(args.device),mask_seq.to(args.device))
                act_seed, bkg_seed, con_seed = utils.select_seed(cas_sigmoid_fuse.detach().cpu(),point_anno.detach().cpu(),step)

            intial_feat_list,num_points=self.cal_featlist(embeded_feature,act_seed,vid_label)
            for c in intial_feat_list.keys():
                if c not in feat_total.keys():
                    feat_total[c]=intial_feat_list[c]
                else:
                    feat_total[c]=torch.cat([feat_total[c],intial_feat_list[c]],dim=0)

        #kmeans for each class
        for c in range(self.class_number):
            if self.proto_num>1:
                center_id, cluster_centers = kmeans(X=feat_total[c], num_clusters=self.proto_num, distance='euclidean', device=args.device)
                # center_id, cluster_centers = kmeans(X=feat_total[c], num_clusters=self.proto_num, distance='cosine', device=args.device)
            else:
                cluster_centers=feat_total[c].mean(dim=0,keepdim=True)
            self.proto_vectors[c]=cluster_centers

    def update_prototype(self,args,feats,act_seq,vid_label):
        self.proto_vectors=self.proto_vectors.to(args.device)
        feat_list,num_points=self.cal_featlist(feats,act_seq,vid_label)

        for c in feat_list.keys():
            if len(feat_list[c])>0:
                feats_proto_distance=self.cal_feat_proto_distance(feat_list[c].unsqueeze(0))

                for b in range(feats_proto_distance.shape[0]):
                    near_d,nearest_idx=feats_proto_distance[b,:,c,:].min(dim=1)
                    
                    for t in range(feat_list[c].shape[0]):
                        match_proto=nearest_idx[t]
                        self.proto_vectors[c,match_proto,:]=self.proto_vectors[c,match_proto,:] * (1 - self.proto_momentum)+self.proto_momentum *feat_list[c][t,:]

    
    def cal_featlist(self,feats,act_seq,vid_label):
        featlist={}
        num_points=torch.zeros([self.class_number])
        for b in range(act_seq.shape[0]):
            gt_class=torch.nonzero(vid_label[b]).cpu().squeeze(1).numpy()

            for c in gt_class:
                act_id=torch.nonzero(act_seq[b,:,c]).squeeze(1)
                num_points[c]+=act_id.shape[0]

                act_feat=feats[b,act_id,:]
                if c not in featlist.keys():
                    featlist[c]=act_feat
                else:
                    featlist[c]=torch.cat(featlist[c],act_feat)

        return featlist,num_points #featlist{c_id:tensor[T,D]}

    def cal_feat_proto_distance(self,feats):
        self.proto_vectors=self.proto_vectors.to(feats.device)
        B,T,D=feats.shape
        N=self.proto_num
        C=self.class_number

        feats_proto_distance=torch.norm(feats.reshape(D,B,T,1,1).expand(-1,-1,-1,C,N)-
                                        self.proto_vectors.reshape(D,1,1,C,N).expand(-1,B,T,-1,-1),2,dim=0)

        # feats_proto_distance=-torch.matmul(feats.reshape(-1,D),self.proto_vectors.reshape(-1,D).permute(1,0))
        # feats_proto_distance=feats_proto_distance.view(B,T,C,N)

        # feats=feats.permute(0,2,1) #B,C,N
        # B,C,T=feats.shape
        # feats_proto_distance=-torch.ones((B, self.class_number,self.proto_num, T)).to(feats.device)
        # for i in range(self.class_number):
        #     self.proto_vectors=self.proto_vectors.to(feats.device)

        #     a=feats-self.proto_vectors[i].reshape(-1,1).expand(-1,T)
        #     b=torch.norm(a,2,dim=1)
        #     feats_proto_distance[:,i,:]=b
        # feats_proto_distance=feats_proto_distance.permute(0,2,1) #B,T,C

        return feats_proto_distance #[B,T,C,N]

    def get_prototype_weight(self,feats):
        feat_proto_distance=self.cal_feat_proto_distance(feats)                                                                              
        feat_nearest_proto_distance,feat_nearest_proto=feat_proto_distance.min(dim=2,keepdim=True)

        feat_proto_distance=feat_proto_distance-feat_nearest_proto_distance
        weight=F.softmax(-feat_proto_distance*self.proto_temperature,dim=2)              

        return weight


if __name__ == '__main__':
    import options
    from dataset import dataset
    args=options.parse_args()
    train_loader=data.DataLoader(dataset(args,phase="train",sample="random"),
                                    batch_size=1,shuffle=True, num_workers=args.num_workers)
    proto=Prototypes(train_loader,args.action_cls_num)
    print(proto.proto_vectors.shape)
    

    
