import torch
import numpy as np
import pandas as pd
import utils
import os
import json
from train import Total_loss
from eval.eval_detection import ANETdetection
from tqdm import tqdm


def test(net, pgnet, args, test_loader, logger, step, test_info, model_file=None, save_file=False, subset='test'):
    with torch.no_grad():
        net.eval()
        pgnet.eval()
        if model_file is not None:
            net.load_state_dict(torch.load(model_file))

        final_result={}
        final_result['version'] = 'VERSION 1.3'
        final_result['results'] = {}
        final_result['external_data'] = {'used': True, 'details': 'Features from I3D Network'}

       
        num_total = 0.
        num_correct = 0.

        load_iter = iter(test_loader)
        for i in tqdm(range(len(test_loader.dataset))):

            _idx, data, vid_label, _point_anno, mask_seq, gt_heatmap, vid_name, vid_len, vid_duration=next(load_iter)
            if data.shape[1]==1:
                continue
            data=data.to(args.device)
            vid_label=vid_label.to(args.device)

            vid_len=vid_len[0].cpu().item()
            vid_duration=vid_duration[0].cpu().item()

            num_segments=data.shape[1]

            # attn_mask=pseudo_label['act_attn_mask']
            # if len(attn_mask.shape)>1:
            #     attn_mask=attn_mask.to(args.device)
            #     vid_score,cas_sigmoid,cas_sigmoid_fuse,embeded_feature,count_feat=net(data,vid_label,attn_mask)
            # else:
            #     vid_score,cas_sigmoid,cas_sigmoid_fuse,_embeded_feature,_=net(data,None)
            vid_score,embeded_feature,cas_sigmoid_fuse,\
                _,_,_=net(data,None,None,None)
            p_heatmap=pgnet(data)
            ###################################
            if save_file:
                # bkg_score=pd.DataFrame((1-p_heatmap[0,:,-1]).cpu().data.numpy())
                bkg_score=pd.DataFrame(cas_sigmoid_fuse[0,:,-1].cpu().data.numpy())
                save_dir=os.path.join(args.output_path,"saved_file","bkg_score")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                bkg_score.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)

                # self_attn=torch.mean(self_attn,dim=-1)
                # self_attn=pd.DataFrame(self_attn[0].cpu().data.numpy())
                # save_dir=os.path.join(args.output_path,"saved_file","self_attn")
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # self_attn.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)

                # cross_attn=torch.mean(cross_attn,dim=-1)
                # cross_attn=pd.DataFrame(cross_attn[0].cpu().data.numpy())
                # save_dir=os.path.join(args.output_path,"saved_file","cross_attn")
                # if not os.path.exists(save_dir):
                #     os.makedirs(save_dir)
                # cross_attn.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)

                

            agnostic_score = 1 - cas_sigmoid_fuse[:,:,-1].unsqueeze(2)
            cas_sigmoid_fuse = cas_sigmoid_fuse[:,:,:-1]

            #video_level test_acc
            label_np = vid_label.cpu().data.numpy()
            score_np = vid_score[0].cpu().data.numpy()
            ###################################
            if save_file:
                score_pd=pd.DataFrame(score_np)
                save_dir=os.path.join(args.output_path,"saved_file","vid_score")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                score_pd.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)

            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < args.class_thresh)] = 0
            pred_np[np.where(score_np >= args.class_thresh)] = 1
            if pred_np.sum() == 0:
                pred_np[np.argmax(score_np)]=1

            correct_pred=np.sum(label_np == pred_np,axis=1)
            num_correct+=np.sum((correct_pred == args.action_cls_num).astype(np.float32))
            num_total+=correct_pred.shape[0]

            #cas after threshold
            cas=cas_sigmoid_fuse
            pred=np.where(score_np >= args.class_thresh)[0]
            if len(pred) == 0:
                pred=np.array([np.argmax(score_np)])
            cas_pred=cas[0].cpu().numpy()[:,pred]
            ###################################
            if save_file:
                cas_pred_pd=pd.DataFrame(cas_pred)
                save_dir=os.path.join(args.output_path,"saved_file","cas_pred")
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cas_pred_pd.to_csv(save_dir+"/"+vid_name[0]+".csv",index=False)
            cas_pred=np.reshape(cas_pred,(num_segments,-1,1))
            cas_pred=utils.upgrade_resolution(cas_pred, args.scale)

            
            #class_agnostic score
            agnostic_score = agnostic_score.expand((-1, -1, args.action_cls_num))
            agnostic_score_np = agnostic_score[0].cpu().data.numpy()[:, pred]
            agnostic_score_np = np.reshape(agnostic_score_np, (num_segments, -1, 1))
            agnostic_score_np = utils.upgrade_resolution(agnostic_score_np, args.scale)

            #GENERATE PROPOSALs
            proposal_dict={}
            for i in range(len(args.act_thresh_cas)):
                cas_temp=cas_pred.copy()

                zero_location = np.where(cas_temp[:,:,0] < args.act_thresh_cas[i])
                cas_temp[zero_location]=0

                seg_list=[]
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:,c,0]>0)
                    seg_list.append(pos)
                
                proposals = utils.get_proposal_oic(args,seg_list,cas_temp,score_np,pred,args.scale,\
                            vid_len,args.frames_per_sec,num_segments,vid_duration,args._lambda,args.gamma)
                for i in range(len(proposals)):
                    class_id=proposals[i][0][0]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id]=[]
                    proposal_dict[class_id]+=proposals[i]

            if args.agnostic_inf:
                for i in range(len(args.act_thresh_agnostic)):
                    cas_temp = cas_pred.copy()

                    agnostic_score_np_temp = agnostic_score_np.copy()

                    zero_location = np.where(agnostic_score_np_temp[:, :, 0] < args.act_thresh_agnostic[i])
                    agnostic_score_np_temp[zero_location] = 0

                    seg_list = []
                    for c in range(len(pred)):
                        pos = np.where(agnostic_score_np_temp[:, c, 0] > 0)
                        seg_list.append(pos)

                    proposals = utils.get_proposal_oic(args, seg_list, cas_temp, score_np, pred, args.scale, \
                                    vid_len, args.frames_per_sec, num_segments,vid_duration,args._lambda,args.gamma)

                    for i in range(len(proposals)):
                        class_id = proposals[i][0][0]
                        if class_id not in proposal_dict.keys():
                            proposal_dict[class_id] = []
                        proposal_dict[class_id] += proposals[i]

            final_proposals=[]
            for class_id in proposal_dict.keys():
                #soft nms
                final_proposals.append(utils.nms(proposal_dict[class_id],thresh=args.nms_thresh,nms_alpha=args.nms_alpha,soft_nms=True))
            final_proposals=[final_proposals[i][j] for i in range(len(final_proposals)) for j in range(len(final_proposals[i]))]
            final_result['results'][vid_name[0]]=utils.result2json(final_proposals,args)

        test_acc=num_correct / num_total

        json_path = os.path.join(args.output_path, 'temp_result_{}.json'.format(subset))
        with open(json_path, 'w') as f:
            json.dump(final_result, f)
        
        tIoU_thresh = args.tIoU_thresh
        gt_path=os.path.join(args.data_path,"gt_full.json")
        if subset=='test':
            log_folder='acc'
            if args.dataset == "THUMOS14":
                subset_name='test'
                if 'full' in gt_path:
                    subset_name='Test'
            elif args.dataset == "ActivityNet13":
                subset_name='val'
            elif args.dataset in ["BEOID",'GTEA']:
                subset_name='validation'
        elif subset=='train':
            log_folder='acc_train'
            subset_name='train'
            if 'full' in gt_path:
                subset_name='Validation'
            if args.dataset in ["BEOID",'GTEA']:
                subset_name='training'

        anet_detection = ANETdetection(gt_path, json_path,
                                   subset=subset_name, tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False,blocked_videos=args.blocked_videos)
        mAP, _ = anet_detection.evaluate()

        
        print("TEST ACC:{:.4f}".format(test_acc))
        if logger is not None:
            logger.log_value('{}/Test accuracy'.format(log_folder), test_acc, step)
            for i in range(tIoU_thresh.shape[0]):
                logger.log_value('{}/mAP@{:.1f}'.format(log_folder,tIoU_thresh[i]), mAP[i], step)

        if args.dataset in ["THUMOS14","BEOID","GTEA"]:
            if logger is not None:
                logger.log_value('{}/Average mAP[0.1:0.7]'.format(log_folder), mAP[:7].mean(), step)
                logger.log_value('{}/Average mAP[0.1:0.5]'.format(log_folder), mAP[:5].mean(), step)
                logger.log_value('{}/Average mAP[0.3:0.7]'.format(log_folder), mAP[2:7].mean(), step)

            test_info["step"].append(step)
            test_info["test_acc"].append(test_acc)
            test_info["average_mAP[0.1:0.7]"].append(mAP[:7].mean())
            test_info["average_mAP[0.1:0.5]"].append(mAP[:5].mean())
            test_info["average_mAP[0.3:0.7]"].append(mAP[2:7].mean())

            for i in range(tIoU_thresh.shape[0]):
                test_info["mAP@{:.1f}".format(tIoU_thresh[i])].append(mAP[i])

            return test_info["average_mAP[0.1:0.7]"][-1]

        elif args.dataset=="ActivityNet13":
            if logger is not None:
                logger.log_value('acc/Average mAP[0.5:0.95]', mAP[:].mean(), step)

            test_info["step"].append(step)
            test_info["test_acc"].append(test_acc)
            test_info["average_mAP[0.5:0.95]"].append(mAP[:].mean())

            for i in range(tIoU_thresh.shape[0]):
                test_info["mAP@{:.2f}".format(tIoU_thresh[i])].append(mAP[i])

            return  test_info["average_mAP[0.5:0.95]"][-1]







            







