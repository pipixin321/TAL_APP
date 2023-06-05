import sys
sys.path.append("./tal_alg/CoRL")
from model import Model
import options
import os
import torch
import pickle
import numpy as np
import utils



def infer_single(feature,duration):
    args=options.parse_args()
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    net=Model(args)

    args.checkpoint=os.path.join(args.model_path,"model_seed_{}.pkl".format(args.seed))
    net.load_state_dict(torch.load(args.checkpoint))
    net=net.to(args.device)

    with torch.no_grad():
        net.eval()
        feature=torch.as_tensor(feature.astype(np.float32))
        data=feature.to(args.device).unsqueeze(0)
        num_segments=data.shape[1]

        vid_score,embeded_feature,cas_sigmoid_fuse,\
                _,_,_=net(data,None,None,None)
        agnostic_score = 1 - cas_sigmoid_fuse[:,:,-1].unsqueeze(2)
        cas_sigmoid_fuse = cas_sigmoid_fuse[:,:,:-1]

        #video_level prediction
        score_np = vid_score[0].cpu().data.numpy()
        pred_np = np.zeros_like(score_np)
        pred_np[np.where(score_np < args.class_thresh)] = 0
        pred_np[np.where(score_np >= args.class_thresh)] = 1
        if pred_np.sum() == 0:
            pred_np[np.argmax(score_np)]=1

        #cas after threshold
        
        cas=cas_sigmoid_fuse
        pred=np.where(score_np >= args.class_thresh)[0]
        if len(pred) == 0:
            pred=np.array([np.argmax(score_np)])
        cas_pred=cas[0].cpu().numpy()[:,pred]
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
                        num_segments,args.frames_per_sec,num_segments,duration,args._lambda,args.gamma)
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
                                num_segments, args.frames_per_sec, num_segments,duration,args._lambda,args.gamma)

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
        results=utils.result2json(final_proposals,args)
        return results


if __name__ == '__main__':
    import cv2
    vid_dir='/mnt/data1/zhx/TAL_APP/tmp/video'
    vids=os.listdir(vid_dir)
    vid=vids[0]
    cap = cv2.VideoCapture(os.path.join(vid_dir,vid))
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    duration=frame_counter/fps


    tmp_feat_dir='/mnt/data1/zhx/TAL_APP/tmp/features'
    files=os.listdir(tmp_feat_dir)
    file=files[0]
    with open(os.path.join(tmp_feat_dir,file),'rb') as f:
        feature=pickle.load(f)
    result=infer_single(feature,duration)
