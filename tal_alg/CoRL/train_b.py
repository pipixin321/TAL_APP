import torch
import torch.nn as nn
import utils
from tqdm import tqdm
import numpy as np

class Total_loss(nn.Module):
    def __init__(self,args):
        super(Total_loss, self).__init__()
        self.lambdas=args.lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')

    def forward(self,args,vid_score,vid_label,cas_sigmoid,point_anno):
        loss={}

        #----------------------------------------loss_vid-----------------------------------------------------#
        loss_vid = self.ce_criterion(vid_score, vid_label)
        loss_vid = loss_vid.mean()
        # loss_vid=-torch.mean(torch.sum(torch.log(vid_score) * vid_label, dim=1))

        #----------------------------------------loss_point---------------------------------------------------#
        loss_frame=0
        loss_frame_bkg=0
        if args.supervision == "point":
            #L_actframe
            point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).to(args.device)), dim=2)
            weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
            num_actions = point_anno.max(dim=2)[0].sum(dim=1)

            focal_weight_act = (1 - cas_sigmoid) * point_anno + cas_sigmoid * (1 - point_anno)
            focal_weight_act = focal_weight_act ** 2
            loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()

            #L_bkgframe
            _, bkg_seed = utils.select_seed(cas_sigmoid.detach().cpu(), point_anno.detach().cpu())
            bkg_seed = bkg_seed.unsqueeze(-1).to(args.device)

            point_anno_bkg = torch.zeros_like(point_anno).to(args.device)
            point_anno_bkg[:,:,-1] = 1

            weighting_seq_bkg = bkg_seed
            num_bkg = bkg_seed.sum(dim=1)

            focal_weight_bkg = (1 - cas_sigmoid) * point_anno_bkg + cas_sigmoid * (1 - point_anno_bkg)
            focal_weight_bkg = focal_weight_bkg ** 2
            loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()
           

        loss_total=self.lambdas[0]*loss_vid+self.lambdas[1]*loss_frame+self.lambdas[2]*loss_frame_bkg

        loss["loss_vid"]=loss_vid
        loss["loss_frame"]=loss_frame
        loss["loss_frame_bkg"]=loss_frame_bkg

        loss["loss_total"]=loss_total
        
        return loss_total, loss


def train(net,args,loader_iter,optimizer,logger,step):
    net.train()
    total_loss={}
    train_num_correct = 0
    train_num_total = 0
    
    for _idx, data, vid_label, point_anno,count, _video_name, _vid_len, _vid_duration in tqdm(loader_iter):

        data=data.to(args.device)
        vid_label=vid_label.to(args.device)
        point_anno=point_anno.to(args.device)
        vid_score,cas_sigmoid,embeded_feature,_=net(data,vid_label)

        criterion=Total_loss(args)
        cost,loss=criterion(args,vid_score,vid_label,cas_sigmoid,point_anno)

        optimizer.zero_grad()
        if not torch.isnan(cost):
            cost.backward()
            optimizer.step()

        with torch.no_grad():
            label_np=vid_label.cpu().numpy()
            score_np=vid_score.cpu().numpy()
            pred_np=np.zeros_like(score_np)
            pred_np[score_np >= args.class_thresh] = 1
            pred_np[score_np < args.class_thresh] = 0
            correct_pred = np.sum(label_np == pred_np, axis=1)

            train_num_correct += np.sum((correct_pred == args.action_cls_num))
            train_num_total += correct_pred.shape[0]

        if not torch.isnan(cost):
            for key in loss.keys():
                if not (key in total_loss.keys()):
                    total_loss[key]=[]
                if loss[key]>0:
                    total_loss[key]+=[loss[key].detach().cpu().item()]
                else:
                    total_loss[key] += [loss[key]]
    
    train_acc = train_num_correct/train_num_total
    print("Train_acc:{:.4f}".format(train_acc))
    for key in total_loss.keys():
        print("{}loss:{:.4f}".format(key,sum(total_loss[key]) / len(loader_iter)))
        logger.log_value("loss/"+key,sum(total_loss[key]) / len(loader_iter), step)
        logger.log_value("acc/train acc",train_acc,step)