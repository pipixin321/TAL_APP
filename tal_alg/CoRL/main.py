import os
import json
import torch
import torch.utils.data as data
from tqdm import tqdm
from tensorboard_logger import Logger
from pseudo_label import pseudo
from prototype import Prototypes
from lr_schedulers import LinearWarmupCosineAnnealingLR

import utils
import options
from model import Model,PGM
from dataset import dataset
from train import train,train_pgm
from test import test

def main(args):
    utils.save_config(args, os.path.join(args.output_path, "config.txt"))
    utils.set_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = device

    pgnet=PGM(args)
    pgnet=pgnet.to(args.device)
    if args.stage==1:
        train_loader=data.DataLoader(dataset(args,phase="train",sample="random",stage=args.stage),
                                            batch_size=1,shuffle=True, num_workers=args.num_workers)
        test_loader=data.DataLoader(dataset(args,phase="test",sample="random",stage=args.stage),
                                            batch_size=1,shuffle=False, num_workers=args.num_workers)
        if args.mode=='train': 
            optimizer=torch.optim.Adam(pgnet.parameters(),lr=args.lr,betas=(0.9,0.999),weight_decay=args.weight_decay)
            for step in tqdm(range(1,args.num_iters+1), total=args.num_iters, dynamic_ncols = True):
                if (step-1) % (len(train_loader)//args.batch_size) == 0:
                    loader_iter=iter(train_loader)
                loss_train=train_pgm(pgnet,args,loader_iter,optimizer)
                if step%50==0:
                    print(loss_train)
                    torch.save(pgnet.state_dict(), os.path.join(args.model_path,"pgnet_seed_{}.pkl".format(args.seed)))
        else:
            pgnet.load_state_dict(torch.load(os.path.join(args.model_path,"pgnet_seed_{}.pkl".format(args.seed))))
            utils.generate_mask(args,pgnet,train_loader,test_loader)



    else:
        net=Model(args)
        if args.checkpoint is not None and os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
            net.load_state_dict(checkpoint)
        net=net.to(device)

        train_loader=data.DataLoader(dataset(args,phase="train",sample="random",stage=args.stage),
                                        batch_size=1,shuffle=True, num_workers=args.num_workers)

        test_loader=data.DataLoader(dataset(args,phase="test",sample="random",stage=args.stage),
                                        batch_size=1,shuffle=False, num_workers=args.num_workers)

        test_info = args.test_info

        proto=Prototypes(train_loader,args.action_cls_num,args.feature_dim)
        proto.init_proto_from_point(args,net)

        optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,betas=(0.9,0.999),weight_decay=args.weight_decay)
        # scheduler = LinearWarmupCosineAnnealingLR(optimizer, 100, 1000)
        scheduler=None
        

        best_mAP=-1
        if args.mode == "train":
            logger=Logger(args.log_path)
            for step in tqdm(range(1,args.num_iters+1), total=args.num_iters, dynamic_ncols = True):
                if (step-1) % (len(train_loader)//args.batch_size) == 0:
                    loader_iter=iter(train_loader)

                
                # if step%50==0:
                #     proto.update_prototype_kmean(args,net,step)
                #     print("#updated prototype>>>>>>>>>>>>>>>>>>>>>>>")
                train(net,proto,args,loader_iter,optimizer,scheduler,logger,step)
                

                if args.pseudo_freq!=0 and step % args.pseudo_freq==0:
                    pseudo(net,proto,args,step,train_loader,cached=True)
                    print("#updated pseudo label>>>>>>>>>>>>>>>>>>>>>>>")

                if step % args.test_iter == 0:
                    # torch.save(net.state_dict(), os.path.join(args.model_path,"checkpoint_model_seed_{}.pkl".format(args.seed)))
                    # train_mAP=test(net,args,train_loader,logger,step,test_info,save_file=False,subset='train')
                    test_mAP=test(net,pgnet,args,test_loader,logger,step,test_info)
                    if test_mAP>best_mAP:
                        best_mAP=test_mAP
                        utils.save_best_record(test_info,os.path.join(args.output_path, "best_record_seed_{}.txt".format(args.seed)),args.dataset)
                        torch.save(net.state_dict(), os.path.join(args.model_path,"model_seed_{}.pkl".format(args.seed)))
                    print("\n Current test_mAP:{:.4f} best_mAP:{:.4f}".format(test_mAP,best_mAP))
                    logger.log_value('acc/best mAP', best_mAP, step)
                    
        else:
            print("Test Start")
            if args.checkpoint=='':
                args.checkpoint=os.path.join(args.model_path,"model_seed_{}.pkl".format(args.seed))
                # args.checkpoint=os.path.join(args.model_path,"checkpoint_model_seed_{}.pkl".format(args.seed))
            net.load_state_dict(torch.load(args.checkpoint))

            # pgnet.load_state_dict(torch.load(os.path.join(args.model_path,"pgnet_seed_{}.pkl".format(args.seed))))
            # pgnet.load_state_dict(torch.load("/root/Weakly_TAL/baseline_v1/ckpt/THUMOS14/models/point_detector/two_stage/pgnet_seed_0.pkl"))
           

            # train_mAP=test(net,pgnet,args,train_loader,None,0,test_info,save_file=True,subset='train')
            # print(test_info)

            test_mAP=test(net,pgnet,args,test_loader,None,0,test_info,save_file=True,subset='test')
            print(test_info)


if __name__ == "__main__":
    hyp = '/mnt/data1/zhx/TAL_APP/tal_alg/CoRL/cfgs/THUMOS14/thumos_i3d.yaml'
    args=options.parse_args(hyp)
    main(args)