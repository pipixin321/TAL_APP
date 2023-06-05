import utils
import torch
import torch.utils.data as data
import os
import numpy as np
import copy

def pseudo(net,proto,args,step,train_loader,cached=False):
    with torch.no_grad():
        net.eval()

        cas_lst = [-1] * len(train_loader.dataset)
        act_seed_lst = [-1] * len(train_loader.dataset)
        bkg_seed_lst = [-1] * len(train_loader.dataset)
        con_seed_lst = [-1] * len(train_loader.dataset)
        act_attn_mask_lst = [-1] * len(train_loader.dataset)
        bkg_attn_mask_lst = [-1] * len(train_loader.dataset)

        matchid_lst = [-1] * len(train_loader.dataset)
        weight_lst = [-1] * len(train_loader.dataset)

        temp_loader = data.DataLoader(
            train_loader.dataset,
            batch_size=1,
            shuffle=True, num_workers=args.num_workers)

        loader_iter = iter(temp_loader)
        
        for i in range(len(train_loader.dataset)):
            _idx, _data, vid_label, point_anno, pseudo_label,gt_count, _video_name, _vid_len, _vid_duration = next(loader_iter)
            

            _data = _data.to(args.device)
            _label = vid_label.to(args.device)
            point_anno = point_anno.to(args.device)
            
            vid_score,embeded_feature,cas_sigmoid_fuse,_,_,_ = net(_data, proto.proto_vectors,_label)
            
            point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(args.device)), dim=2)   
            act_seed, bkg_seed,con_seed = utils.select_seed(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu(),step)
            # num_con=con_seed.max(dim=2)[0].sum(dim=1)
            # if num_con>0:
            #     print(num_con)

            act_seed_1d=act_seed.max(dim=2)[0]
            act_attn_mask=torch.matmul(act_seed_1d.permute(1,0),act_seed_1d)
            bkg_attn_mask=torch.matmul(bkg_seed.permute(1,0),bkg_seed)

            cas_lst[_idx[0]] = cas_sigmoid_fuse[0].detach().cpu()
            act_seed_lst[_idx[0]] = act_seed[0].detach().cpu()
            bkg_seed_lst[_idx[0]] = bkg_seed[0].detach().cpu()
            con_seed_lst[_idx[0]] = con_seed[0].detach().cpu()

            act_attn_mask_lst[_idx[0]]=act_attn_mask.detach().cpu()
            bkg_attn_mask_lst[_idx[0]]=bkg_attn_mask.detach().cpu()

            #save match result
            import torch.nn.functional as F
            feat_distance=proto.cal_feat_proto_distance(embeded_feature)#[B,T,C,N]
            feat_nearest_proto_distance,feat_nearest_proto=feat_distance.min(dim=3,keepdim=True)
            feat_proto_distance=feat_distance-feat_nearest_proto_distance
            weight=F.softmax(-feat_proto_distance,dim=3)  #[B,T,C,N]
            
            feat_nearest_proto=feat_nearest_proto.squeeze(-1)[0].detach().cpu()
            weight=weight.max(dim=-1)[0][0].detach().cpu()
            matchid_lst[_idx[0]]=feat_nearest_proto
            weight_lst[_idx[0]]=weight
            
    
        # train_loader.dataset.pseudo_label['act_seq'] = copy.deepcopy(act_seed_lst)
        # train_loader.dataset.pseudo_label['bkg_seq'] = copy.deepcopy(bkg_seed_lst)
        # train_loader.dataset.pseudo_label['act_attn_mask'] = copy.deepcopy(act_attn_mask_lst)
        # train_loader.dataset.pseudo_label['bkg_attn_mask'] = copy.deepcopy(bkg_attn_mask_lst)

        if cached:# and step%50==0:
            savedir=os.path.join(args.output_path,"pseudo_label")
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            # act_seed_lst=train_loader.dataset.pseudo_label['act_seq']
            # np.save(os.path.join(savedir,'act_step{}.npy'.format(step)),act_seed_lst)

            cas_lst=np.array([cas.data.numpy() for cas in cas_lst],dtype=object)
            # np.save(os.path.join(savedir,'cas_step{}.npy'.format(step)),cas_lst)
            # act_seed_lst=np.array([act_seed.data.numpy() for act_seed in act_seed_lst],dtype=object)
            np.save(os.path.join(savedir,'act_step{}.npy'.format(step)),act_seed_lst)
            bkg_seed_lst=np.array([bkg_seed.data.numpy() for bkg_seed in bkg_seed_lst],dtype=object)
            np.save(os.path.join(savedir,'bkg_step{}.npy'.format(step)),bkg_seed_lst)
            con_seed_lst=np.array([con_seed.data.numpy() for con_seed in con_seed_lst],dtype=object)
            np.save(os.path.join(savedir,'con_step{}.npy'.format(step)),con_seed_lst)

            matchid_lst=np.array([matchid.data.numpy() for matchid in matchid_lst],dtype=object)
            np.save(os.path.join(savedir,'matchid_step{}.npy'.format(step)),matchid_lst)
            weight_lst=np.array([weight.data.numpy() for weight in weight_lst],dtype=object)
            np.save(os.path.join(savedir,'weight_step{}.npy'.format(step)),weight_lst)

            
    return