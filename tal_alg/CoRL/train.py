import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

def get_feature_loss(args,embeded_feature,act_seq,vid_label):
        act_seq_diff=act_seq[:,1:]-act_seq[:,:-1]
        loss_feat=0
        for b in range(act_seq.shape[0]):
            gt_class=torch.nonzero(vid_label[b]).squeeze(1)
            loss_feat_batch=0
            act_count=0

            for c in gt_class:
                range_idx=torch.nonzero(act_seq_diff[b,:,c]).squeeze(1)
                range_idx=range_idx.cpu().data.numpy().tolist()
                if type(range_idx) is not list:
                    range_idx = [range_idx]
                if len(range_idx) == 0:
                    continue
                if act_seq_diff[b, range_idx[0], c] != 1:
                    range_idx = [-1] + range_idx 
                if act_seq_diff[b, range_idx[-1], c] != -1:
                    range_idx = range_idx + [act_seq_diff.shape[1] - 1]

                label_lst = []
                feature_lsts = []

                # bkg
                if range_idx[0]>-1:
                    start_bkg=0
                    end_bkg=range_idx[0]
                    bkg_len=end_bkg-start_bkg+1
                    label_lst.append(0)
                    feature_lsts.append(utils.feature_sampling(args,embeded_feature[b],start_bkg, end_bkg + 1))

                # act
                for i in range(len(range_idx)//2):
                    if range_idx[2*i+1]-range_idx[2*i]<1:
                        continue
                    label_lst.append(1)
                    feature_lsts.append(utils.feature_sampling(args,embeded_feature[b],range_idx[2*i] + 1, range_idx[2*i + 1] + 1))

                    if range_idx[2*i+1]!=act_seq_diff.shape[1]-1:
                        start_bkg=range_idx[2*i + 1] + 1
                        if i == len(range_idx)//2-1:#act bkg -
                            end_bkg=act_seq_diff.shape[1]-1
                        else:#act bkg act
                            end_bkg=range_idx[2*i+2]
                        label_lst.append(0)
                        feature_lsts.append(utils.feature_sampling(args,embeded_feature[b],start_bkg,end_bkg+1))
                    act_count+=1

                if sum(label_lst)>1:
                    # for n in range(3):
                    #     feature_lst=[f[n] for f in feature_lsts]
                    feature_lst = torch.stack(feature_lsts, 0).clone()
                    feature_lst = feature_lst / torch.norm(feature_lst, dim=1, p=2).unsqueeze(1)
                    label_lst = torch.tensor(label_lst).to(args.device).float()
                    sim_matrix = torch.matmul(feature_lst, torch.transpose(feature_lst, 0, 1)) / 0.1
                    sim_matrix = torch.exp(sim_matrix)
                    sim_matrix = sim_matrix.clone().fill_diagonal_(0)
                    scores = (sim_matrix * label_lst.unsqueeze(1)).sum(dim=0) / sim_matrix.sum(dim=0)
                    loss_feat_batch += (-label_lst * torch.log(scores)).sum() / label_lst.sum()
                            
            if act_count>0:
                loss_feat+=loss_feat_batch

        loss_feat=loss_feat/act_seq.shape[0]
        return loss_feat

def COUNTINGLOSS(features, gt_count, seq_len, device):
    ''' features: torch tensor dimension (B, n_element, n_class),
        gt_count: torch tensor dimension (B, n_class) of integer value,
        seq_len: numpy array of dimension (B,) indicating the length of each video in the batch, 
        return: torch tensor of dimension 0 (value) '''

    pos_loss, neg_loss, num = 0, 0, 0
    inv_gt_count = (gt_count > 0).float() / (gt_count + 1e-10)
    for i in range(features.size(0)):
        # categories present in video
        mask_pos = (gt_count[i]<int(seq_len[i])) * (gt_count[i]>0)
        # categories absent
        mask_neg = (mask_pos==0)
        pred_count = (features[i,:seq_len[i]]).sum(0)
        pos_loss += ((pred_count[mask_pos] - Variable(gt_count[i][mask_pos],requires_grad=False)) * inv_gt_count[i][mask_pos]).abs().sum() # relative L1 
        neg_loss += 0.001* pred_count[mask_neg==1].abs().sum()
        num += 1
    if num > 0:
        return (pos_loss+neg_loss)/num
    else:
        return torch.zeros(1).to(device)

def get_feature(seq,embeded_feature):#seq:[T]
    seq_diff=seq[1:]-seq[:-1]
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

    feature_lsts = []
    idx=[]
    for i in range(len(range_idx)//2):
        if range_idx[2*i+1]-range_idx[2*i]<1:
            continue
        feature_lsts.append(embeded_feature[range_idx[2*i] + 1: range_idx[2*i + 1] + 1].clone())
        idx.append([range_idx[2*i] + 1,range_idx[2*i + 1] + 1])

    return feature_lsts


def KL_loss(pred,target,vid_label):
    kl_loss_total=0

    for b in range(pred.shape[0]):
        # kl_loss_func = nn.KLDivLoss(reduction="sum")#batchmean

        # pred_bkg_score=pred[b,:,-1]
        # tgt_bkg_score=target[b,:,-1]
        # bkg_kl_loss=kl_loss_func(F.log_softmax(pred_bkg_score,dim=0),F.softmax(tgt_bkg_score,dim=0))
        # # bkg_kl_loss2=kl_loss_func(pred_bkg_score,tgt_bkg_score)
        # act_kl_loss=kl_loss_func(F.log_softmax(pred[b],dim=0),F.softmax(target[b],dim=0))

        ce_criterion = nn.BCELoss(reduction='none')
        ce_loss=ce_criterion(pred[b],target[b].detach()).mean()

        kl_loss_total+=ce_loss

    
    kl_loss=kl_loss_total/pred.shape[0]

    return kl_loss


def contrastive_loss(args,proto,embeded_feature,act_seed,bkg_seed,vid_label): #proto:[C,N,D]
    loss_contra=0
    proto_vectors=proto.proto_vectors.clone().to(embeded_feature.device)
    proto_vectors_norm=proto_vectors/torch.norm(proto_vectors,dim=2,p=2).unsqueeze(2)
    
    intra_sim_matrix=torch.matmul(proto_vectors_norm,torch.transpose(proto_vectors_norm,1,2))
    loss_intra_proto=intra_sim_matrix.mean()

    for b in range(act_seed.shape[0]):
        gt_class=torch.nonzero(vid_label[b]).squeeze(1)
        act_featlst=[]
        for c in gt_class:
            act_featlst.append(get_feature(act_seed[b,:,c],embeded_feature[b,:,:].clone()))
        
        #bkg_act contra
        bkg_feat=get_feature(bkg_seed[b].squeeze(-1),embeded_feature[b,:,:].clone())
        if len(bkg_feat)==0:
            continue
        b_feat=torch.cat(bkg_feat,0).clone()
        b_feat = b_feat / torch.norm(b_feat, dim=1, p=2).unsqueeze(1)
        b_sim_matrix=torch.matmul(b_feat.unsqueeze(0).expand(args.action_cls_num,-1,-1), torch.transpose(proto_vectors_norm, 1, 2)) / 0.1
        b_sim_matrix = torch.exp(b_sim_matrix)


        #video_level contra
        for c_id,c_feat_lst in enumerate(act_featlst):
            if c_feat_lst is not None:
                if len(c_feat_lst)>0:
                    c=gt_class[c_id]
                    c_feat=torch.cat(c_feat_lst,0).clone()
                    c_feat = c_feat / torch.norm(c_feat, dim=1, p=2).unsqueeze(1)

                    a_sim_matrix = torch.matmul(c_feat.unsqueeze(0).expand(args.action_cls_num,-1,-1), torch.transpose(proto_vectors_norm, 1, 2)) / 0.1
                    a_sim_matrix = torch.exp(a_sim_matrix)
                    loss_contra_proto =  - torch.log(a_sim_matrix[c].sum().sum()/a_sim_matrix.sum().sum().sum())
                    # loss_contra_proto =  - torch.log(a_sim_matrix[c].max(dim=-1)[0].sum()/a_sim_matrix.sum())

                    loss_contra_bkg =  - torch.log(a_sim_matrix[c].mean().sum()/(a_sim_matrix[c].mean().sum()+b_sim_matrix[c].mean().sum()))
                    
                    lambda_bkg=0.5
                    loss_contra += (1-lambda_bkg) * loss_contra_proto + lambda_bkg * loss_contra_bkg


                    # temporal structure loss
                    # L=5
                    # if len(c_feat_lst)>1:
                    #     sampled_feat_lst=[]
                    #     for inst_feat in c_feat_lst:
                    #         sampled_feat_lst.append(utils.feature_sampling(args,inst_feat,0,inst_feat.shape[0],L,'mean'))

                    #     sampled_feat_total=torch.cat(sampled_feat_lst,0).clone()
                    #     sampled_feat_total = sampled_feat_total / torch.norm(sampled_feat_total, dim=1, p=2).unsqueeze(1)
                    #     c_sim_matrix=torch.matmul(sampled_feat_total,torch.transpose(proto_vectors_norm[c],0,1))
                    #     match_score=F.softmax(c_sim_matrix*10,dim=1)  #temperture

                    #     num_pair=0
                    #     for inst_id in range(len(sampled_feat_lst)):
                    #         for inst_jd in range(inst_id+1,len(sampled_feat_lst)):
                    #             score_distance=torch.norm(match_score[inst_id*L:(inst_id+1)*L]-match_score[inst_jd*L:(inst_jd+1)*L],2,dim=1)
                    #             loss_temporal += score_distance.mean()

                    #             num_pair += 1
                    #     loss_temporal = loss_temporal / num_pair

        loss_contra=loss_contra/gt_class.shape[0]

    loss_contra=loss_contra/act_seed.shape[0]


    return loss_contra,loss_intra_proto

def base_loss(args,act_seed,bkg_seed,vid_score,vid_label,cas_sigmoid_fuse,point_anno):
    loss_frame=0
    loss_frame_bkg=0
    #video-level loss
    ce_criterion = nn.BCELoss(reduction='none')
    loss_vid = ce_criterion(vid_score, vid_label)
    loss_vid = loss_vid.mean()

    #frame-level loss
    #gt frame loss]
    act_seed=act_seed.to(args.device)
    focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
    focal_weight_act = focal_weight_act ** 2
    weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
    num_actions = point_anno.max(dim=2)[0].sum(dim=1)
    if num_actions>0:
        loss_frame = (((focal_weight_act * ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()
    

    #bkg frame loss
    bkg_seed = bkg_seed.unsqueeze(-1).to(args.device)
    point_anno_bkg = torch.zeros_like(point_anno).to(args.device)
    point_anno_bkg[:,:,-1] = 1

    weighting_seq_bkg = bkg_seed
    num_bkg = bkg_seed.sum(dim=1)
    focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
    focal_weight_bkg = focal_weight_bkg ** 2
    if num_bkg>0:
        loss_frame_bkg = (((focal_weight_bkg * ce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()

    # #pseudo frame loss
    # act_seed = torch.cat((act_seed, torch.zeros((act_seed.shape[0], act_seed.shape[1], 1)).to(args.device)), dim=2)
    # focal_weight_pse_act = (1 - cas_sigmoid_fuse) * act_seed + cas_sigmoid_fuse * (1 - act_seed)
    # focal_weight_pse_act = focal_weight_pse_act ** 2
    # weighting_seq_pse_act = act_seed.max(dim=2, keepdim=True)[0].to(args.device)-weighting_seq_act
    # num_pse_actions=act_seed.max(dim=2)[0].sum(dim=1)-num_actions
    # if num_pse_actions>0:
    #     loss_pseudo_frame = (((focal_weight_pse_act * ce_criterion(cas_sigmoid_fuse, act_seed) * weighting_seq_pse_act).sum(dim=2)).sum(dim=1) / num_pse_actions).mean()
    #     loss_frame+=loss_pseudo_frame
    
    # #context_frame_loss
    # con_seed=con_seed.to(args.device)
    # focal_weight_con=(1-cas_sigmoid_fuse)*con_seed+cas_sigmoid_fuse*(1-con_seed)
    # focal_weight_con=focal_weight_con**2
    # weighting_seq_con = con_seed.max(dim=2, keepdim=True)[0].to(args.device)

    # num_con=con_seed.max(dim=2)[0].sum(dim=1)
    # if num_con>0:
    #     loss_con_frame=(((focal_weight_con * self.ce_criterion(cas_sigmoid_fuse, con_seed) * weighting_seq_con).sum(dim=2)).sum(dim=1) / num_con).mean()
    
    # if step%10==0:
    #     print('numbers: gt:{} pse_act:{} context:{} bkg:{}'.format(num_actions.cpu().item(),num_pse_actions.cpu().item(),num_con.cpu().item(),num_bkg.cpu().item()))


    return loss_vid,loss_frame,loss_frame_bkg


class Total_loss(nn.Module):
    def __init__(self,args):
        super(Total_loss, self).__init__()
        self.lambdas=args.lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')

    def forward(self,args,step,vid_label,point_anno,proto,
                vid_score,cas_sigmoid_fuse,embeded_feature,
                vid_score_comp,embeded_feature_comp,cas_sigmoid_fuse_comp):
        loss={}

        #----------------------------------------base_loss---------------------------------------------------#
        loss_frame=0
        loss_frame_bkg=0
        loss_pseudo_frame=0

        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(args.device)), dim=2)
        #select seed
        act_seed, bkg_seed, con_seed = utils.select_seed(cas_sigmoid_fuse.detach().cpu(),point_anno.detach().cpu(),step)
        # bkg_seed=pseudo_label['bkg_seq']
        # act_seed=pseudo_label['act_seq']


        loss_vid,loss_frame,loss_frame_bkg=base_loss(args,act_seed,bkg_seed,vid_score,vid_label,cas_sigmoid_fuse,point_anno)
        loss["loss_vid"]=loss_vid
        loss["loss_frame"]=loss_frame
        loss["loss_frame_bkg"]=loss_frame_bkg
        loss["loss_pseudo_frame"]=loss_pseudo_frame

        #----------------------------------------contrastive_loss---------------------------------------------------#
        loss_contrastive=0
        if bkg_seed.sum(dim=1)>0:
            loss_contrastive,loss_intra_proto=contrastive_loss(args,proto,embeded_feature,act_seed,bkg_seed,vid_label)
            loss["loss_contrastive"]=loss_contrastive
            loss["loss_intra_proto"]=loss_intra_proto
            proto.update_prototype(args,embeded_feature.detach(),act_seed,vid_label)

        #----------------------------------------kl_loss---------------------------------------------------#
        loss_kl=0
        # loss_kl=KL_loss(cas_sigmoid_fuse,cas_sigmoid_fuse_comp,vid_label)
        # loss['kl_loss']=loss_kl

        #----------------------------------------loss_count---------------------------------------------------#
        loss_count=0
        # countloss_mult = 1 if args.dataset=='THUMOS14' else 0.1
        # seq_len = torch.tensor(count_feat.shape[1]*torch.ones(count_feat.shape[0]),dtype=torch.int64)
        # if args.count_loss:
        #     loss_count=COUNTINGLOSS(count_feat, gt_count, seq_len, args.device)*countloss_mult
        #     loss["loss_count"]=loss_count

        #----------------------------------------loss_feat---------------------------------------------------#  
        loss_feat=0    
        # loss_feat=get_feature_loss(args,embeded_feature,act_seed,vid_label)
        # loss["loss_feat"]=loss_feat

        
        loss_total=self.lambdas[0]*loss_vid+self.lambdas[1]*loss_frame+self.lambdas[2]*loss_frame_bkg+self.lambdas[3]*loss_contrastive+\
                        1.0*loss_kl+1.0*loss_feat+0.5*loss_pseudo_frame
        loss["loss_total"]=loss_total
        
        return loss_total, loss


def train(net,proto,args,loader_iter,optimizer,scheduler,logger,step):
        net.train()
        total_loss={}
        total_cost=[]
        optimizer.zero_grad()

        for batch in range(args.batch_size):

            _idx, data, vid_label, point_anno, mask_seq, gt_heatmap, _video_name, _vid_len, _vid_duration=next(loader_iter)

            data=data.to(args.device)
            vid_label=vid_label.to(args.device)
            point_anno=point_anno.to(args.device)
            gt_heatmap=gt_heatmap.to(args.device)

            # attn_mask=pseudo_label['act_attn_mask']
            # if len(attn_mask.shape)>1:
            #     attn_mask=attn_mask.to(args.device)
            #     vid_score,cas_sigmoid,cas_sigmoid_fuse,embeded_feature,count_feat=net(data,proto.proto_vectors,vid_label,attn_mask)
            # else:
            #     vid_score,cas_sigmoid,cas_sigmoid_fuse,embeded_feature,count_feat=net(data, proto.proto_vectors, vid_label)
            # criterion=Total_loss(args)
            # cost,loss=criterion(args,step,vid_score,vid_label,cas_sigmoid,cas_sigmoid_fuse,point_anno,
            #                         pseudo_label,embeded_feature,count_feat,gt_count,proto)
            # total_cost.append(cost)


            vid_score_base,embeded_feature_base,cas_sigmoid_fuse_base,\
                vid_score_comp,embeded_feature_comp,cas_sigmoid_fuse_comp=net(data, proto.proto_vectors, vid_label, None)
            criterion=Total_loss(args)
            cost,loss=criterion(args,step,vid_label,point_anno,proto,
                                vid_score_base,cas_sigmoid_fuse_base,embeded_feature_base,
                                vid_score_comp,embeded_feature_comp,cas_sigmoid_fuse_comp)
            total_cost.append(cost)

            if not torch.isnan(cost):
                for key in loss.keys():
                    if not (key in total_loss.keys()):
                        total_loss[key]=[]

                    if loss[key]>0:
                        total_loss[key]+=[loss[key].detach().cpu().item()]
                    else:
                        total_loss[key] += [loss[key]]
        
        total_cost=sum(total_cost) / args.batch_size
        total_cost.backward(retain_graph=True) #retain_graph=True for contrastive loss
        optimizer.step()
            
        for key in total_loss.keys():
            # print("loss/{}:{}".format(key,sum(total_loss[key]) / args.batch_size))
            logger.log_value("loss/"+key,sum(total_loss[key]) / args.batch_size, step)

        if scheduler is not None:
            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            logger.log_value("lr",lr, step)


def heatmap_loss(args,p_heatmap,gt_heatmap):
    ce_criterion = nn.BCELoss(reduction='none')
    a=ce_criterion(p_heatmap,gt_heatmap)
    mask_neg=torch.zeros((gt_heatmap.shape[0],gt_heatmap.shape[1],1)).to(args.device)
    mask_neg[gt_heatmap<0.1]=1
    num_neg=mask_neg.sum()
    mask_pos=1-mask_neg
    num_pos=mask_pos.sum()
    loss_heatmap_neg=((a*mask_neg).sum(dim=2).sum(dim=1)/num_neg).mean()
    loss_heatmap_pos=((a*mask_pos).sum(dim=2).sum(dim=1)/num_pos).mean()
    loss_heatmap=loss_heatmap_neg+loss_heatmap_pos
    # loss_norm=torch.norm(p_heatmap,p=2)
    # loss_heatmap+=0.1*loss_norm
    return loss_heatmap

def train_pgm(net,args,loader,optimizer):
    net.train()
    total_cost=[]
    optimizer.zero_grad()
    for batch in range(args.batch_size):
        _idx, data, vid_label, point_anno, pseudo_label, gt_heatmap, _video_name, _vid_len, _vid_duration=next(loader)
        data=data.to(args.device)
        gt_heatmap=gt_heatmap.to(args.device)
        p_heatmap=net(data)
        loss_heatmap=heatmap_loss(args,p_heatmap,gt_heatmap)
        total_cost.append(loss_heatmap)
    total_cost=sum(total_cost) / args.batch_size
    total_cost.backward()
    optimizer.step()
    return total_cost





    


