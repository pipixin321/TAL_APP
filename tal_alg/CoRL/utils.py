import glob
import os
import torch 
import random 
import numpy as np 
import torch.nn as nn 
from scipy.interpolate import interp1d

def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False #set True to find optimal convolution for each layer   default:True
    torch.backends.cudnn.deterministic = True #set True to get same result while given same input and same model   default:False


def check_file(file):
    # Searches for file if not found locally
    if os.path.isfile(file):
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        return files[0]  # return first file if multiple found


def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()

def save_best_record(test_info, file_path, dataset):
    if dataset in ["THUMOS14","BEOID","GTEA"]:
        fo = open(file_path, "w")
        fo.write("Step: {}\n".format(test_info["step"][-1]))
        fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
        fo.write("average_mAP[0.1:0.7]: {:.4f}\n".format(test_info["average_mAP[0.1:0.7]"][-1]))
        fo.write("average_mAP[0.1:0.5]: {:.4f}\n".format(test_info["average_mAP[0.1:0.5]"][-1]))
        fo.write("average_mAP[0.3:0.7]: {:.4f}\n".format(test_info["average_mAP[0.3:0.7]"][-1]))
        
        tIoU_thresh = np.linspace(0.1, 0.7, 7)
        for i in range(len(tIoU_thresh)):
            fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))
        fo.close()

    elif dataset=="ActivityNet13":
        fo = open(file_path, "w")
        fo.write("Step: {}\n".format(test_info["step"][-1]))
        fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
        fo.write("average_mAP[0.5:0.95]: {:.4f}\n".format(test_info["average_mAP[0.5:0.95]"][-1]))

        tIoU_thresh=np.arange(0.50, 1.00, 0.05)
        for i in range(len(tIoU_thresh)):
            fo.write("mAP@{:.2f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.2f}".format(tIoU_thresh[i])][-1]))
        fo.close()

def feature_sampling(args,features, start, end, num_divide=3, method='random'):
    step = (end - start) / num_divide

    feature_lst = torch.zeros((num_divide, features.shape[1])).to(args.device)
    for i in range(num_divide):
        start_point = int(start + step * i)
        end_point = int(start + step * (i+1))
        
        if start_point >= end_point:
            end_point += 1

        if method=='random':
            sample_id = np.random.randint(start_point, end_point)
            feat_i=features[sample_id]
        elif method == 'mean':
            feat_i=features[start_point:end_point].mean(dim=0)
        else:
            print("Unvalid method")
            

        feature_lst[i] = feat_i

    if method == 'random':
        return feature_lst.mean(dim=0)
    else:
        return feature_lst

def upgrade_resolution(arr, scale):
    
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale

def get_proposal_oic(args,tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, v_duration, _lambda=0.25, gamma=0.2):#0.25 0.2
    # t_factor = float(16 * v_len) / (scale * num_segments * sampling_frames) if args.dataset=='THUMOS14' else v_duration/(scale * num_segments)
    t_factor = v_duration/(scale * num_segments)
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

def nms(proposals, thresh, nms_alpha=0.2,soft_nms=False):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]
    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]
    keep = []
    not_keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        if soft_nms:
            inv_inds = np.where(iou >= thresh)[0]
            props_mod = proposals[order[inv_inds + 1]]
            for k in range(props_mod.shape[0]):
                props_mod[k, 1] = props_mod[k, 1] * np.exp(-np.square(iou[inv_inds][k]) / nms_alpha)

            not_keep.extend(props_mod.tolist())
        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]
    if soft_nms:
        keep.extend(not_keep)
    return keep

def result2json(result,args):
    result_file = []    
    for i in range(len(result)):
        line = {'label': args.class_name_lst[int(result[i][0])], 'score': result[i][1],
                'segment': [result[i][2], result[i][3]]}
        result_file.append(line)
    return result_file

def select_seed(cas_sigmoid_fuse, point_anno,step):
    point_anno_agnostic = point_anno.max(dim=2)[0]
    bkg_seed = torch.zeros_like(point_anno_agnostic)
    act_seed = point_anno.clone().detach()

    act_thresh = 0.1 #0.1
    bkg_thresh = 0.95
    psuedo_act_ratio=0.9 # psuedo_act_ratio=0.99-(0.99-0.95)*step/500

    agnostic_pseudo=True
    # cas_sigmoid_fuse[:,:,:-1]=cas_sigmoid_fuse[:,:,:-1]*weight
    # if step<300:
    #     act_thresh = 0.01+(0.05-0.01)*step/300 #(DYPS)
    # else:
    #     act_thresh=0.05


    bkg_score = cas_sigmoid_fuse[:,:,-1]
    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1)
        if len(act_idx)==0:
            # print("no point label")
            continue

        """ most left """
        if act_idx[0] > 0:
            bkg_score_tmp = bkg_score[b,:act_idx[0]]
            idx_tmp = bkg_seed[b,:act_idx[0]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[:start_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[:max_index+1] = 1

            """ pseudo action point selection """
            if agnostic_pseudo:
                for j in range(act_idx[0] - 1, -1, -1):
                    if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                        act_seed[b, j] = act_seed[b, act_idx[0]]
                    else:
                        break

            else:
                gt_class=torch.nonzero(point_anno[:,act_idx[0],:].squeeze(0)).squeeze(1)
                for c in gt_class:
                    for j in range(act_idx[0] - 1, -1, -1):
                        if cas_sigmoid_fuse[b,j,c]>=psuedo_act_ratio and bkg_seed[b][j] < 1:
                            act_seed[b, j, c]=1
                        else:
                            break

        """ most right """
        if act_idx[-1] < (point_anno.shape[1] - 1):
            bkg_score_tmp = bkg_score[b,act_idx[-1]+1:]
            idx_tmp = bkg_seed[b,act_idx[-1]+1:]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                idx_tmp[start_index:] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index:] = 1

            """ pseudo action point selection """
            if agnostic_pseudo:
                for j in range(act_idx[-1] + 1, point_anno.shape[1]):
                    if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                        act_seed[b, j] = act_seed[b, act_idx[-1]]
                    else:
                        break

            else:
                gt_class=torch.nonzero(point_anno[:,act_idx[0],:].squeeze(0)).squeeze(1)
                for c in gt_class:
                    for j in range(act_idx[-1] + 1, point_anno.shape[1]):
                        if cas_sigmoid_fuse[b,j,c]>=psuedo_act_ratio and bkg_seed[b][j] < 1:
                            act_seed[b, j, c]=1
                        else:
                            break
            
        """ between two instances """
        for i in range(len(act_idx) - 1):
            if act_idx[i+1] - act_idx[i] <= 1:
                continue

            bkg_score_tmp = bkg_score[b,act_idx[i]+1:act_idx[i+1]]
            idx_tmp = bkg_seed[b,act_idx[i]+1:act_idx[i+1]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1

            if idx_tmp.sum() >= 2:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                end_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[start_index+1:end_index] = 1                                   
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index] = 1

            """ pseudo action point selection """
            if agnostic_pseudo:
                for j in range(act_idx[i] + 1, act_idx[i+1]):
                    if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                        act_seed[b, j] = act_seed[b, act_idx[i]]
                    else:
                        break
                for j in range(act_idx[i+1] - 1, act_idx[i], -1):
                    if bkg_score[b][j] <= act_thresh and bkg_seed[b][j] < 1:
                        act_seed[b, j] = act_seed[b, act_idx[i+1]]
                    else:
                        break
            else:
                gt_class=torch.nonzero(point_anno[:,act_idx[0],:].squeeze(0)).squeeze(1)
                for c in gt_class:
                    for j in range(act_idx[i] + 1, act_idx[i+1]):
                        if cas_sigmoid_fuse[b,j,c]>=psuedo_act_ratio and bkg_seed[b][j] < 1:
                            act_seed[b, j, c]=1
                        else:
                            break
                for c in gt_class:
                    for j in range(act_idx[i+1] - 1, act_idx[i], -1):
                        if cas_sigmoid_fuse[b,j,c]>=psuedo_act_ratio and bkg_seed[b][j] < 1:
                            act_seed[b, j, c]=1
                        else:
                            break

    con_seed = torch.zeros_like(point_anno)
    # con_thresh=0.9
    # gt_class=torch.nonzero(point_anno.max(dim=1)[0].squeeze(0)).squeeze(1)
    # for c in gt_class:
    #     for b in  range(point_anno.shape[0]):
    #         bkg_idx=torch.nonzero(bkg_seed[b]).squeeze(1)
    #         for idx in bkg_idx:
    #             # if bkg_score[b,idx]<=con_thresh:
    #             if cas_sigmoid[b,idx,c]>=con_thresh:
    #                 # print(idx,bkg_score[b,idx])
    #                 con_seed[b,idx,c]=1
    #                 con_seed[b,idx,-1]=1
    #                 # bkg_seed[b,idx]=0

    
    return act_seed, bkg_seed, con_seed

def generate_single(args,vid_name,data,pgnet,th1=0.5,th2=0.1):
    data=data.to(args.device)
    p_heatmap=pgnet(data).detach().cpu()
    p_heatmap=p_heatmap[0].squeeze(1)
    num_seg=len(p_heatmap)

    point_bins=[]
    for i in range(1,num_seg-1):
        if p_heatmap[i]>p_heatmap[i-1] and p_heatmap[i]>p_heatmap[i+1] and p_heatmap[i]>th1:
            point_bins.append(i)
    num_bins=len(point_bins)

    mask_seq=np.zeros(num_seg)
    mask_seq[p_heatmap>th2]=1

    print(mask_seq.sum(),num_seg)
    savedir=os.path.join(args.data_path,'mask')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    np.save(os.path.join(savedir,'{}.npy'.format(vid_name[0])),mask_seq)
            


def generate_mask(args,pgnet,train_loader,test_loader):
    from tqdm import tqdm
    print("mask generate")
    with torch.no_grad():
        pgnet.eval()

        load_iter = iter(train_loader)
        for i in tqdm(range(len(train_loader.dataset))):
            _idx, data, vid_label, _point_anno, _, gt_heatmap, vid_name, vid_len, vid_duration=next(load_iter)
            generate_single(args,vid_name,data,pgnet)

        load_iter = iter(test_loader)
        for i in tqdm(range(len(test_loader.dataset))):
            _idx, data, vid_label, _point_anno, _, gt_heatmap, vid_name, vid_len, vid_duration=next(load_iter)
            generate_single(args,vid_name,data,pgnet)
            



