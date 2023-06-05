import os
import json
import torch
import torch.utils.data as data
from tqdm import tqdm
from tensorboard_logger import Logger

import utils
import options
from model import Model
from dataset import dataset
from train_b import train
from test import test

if __name__ == "__main__":
    import time
    t_start =time.perf_counter() 
    
    args=options.parse_args()
    utils.save_config(args, os.path.join(args.output_path, "config.txt"))
    utils.set_seed(args.seed)

    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = device

    net=Model(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint)
    net=net.to(device)

    train_loader=data.DataLoader(dataset(args,phase="train",sample="random"),
                                    batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,drop_last=False)

    test_loader=data.DataLoader(dataset(args,phase="test",sample="random"),
                                    batch_size=1,shuffle=False, num_workers=args.num_workers,drop_last=False)

    test_info = args.test_info
    optimizer=torch.optim.Adam(net.parameters(),lr=args.lr,betas=(0.9,0.999),weight_decay=args.weight_decay)
    logger=Logger(args.log_path)

    best_mAP=-1
    if args.dataset == 'THUMOS14':
        epochs=500
        test_epoch=2
    else:
        epochs=500
        test_epoch=5
    if args.mode == "train":
        for step in tqdm(range(1,epochs+1), total=epochs, dynamic_ncols = True):
            # if step/test_epoch==2:
            #     test_epoch=max(5,int(test_epoch/2))
            train(net,args,train_loader,optimizer,logger,step)
            if step % test_epoch == 0:
                test_mAP=test(net,args,test_loader,logger,step,test_info)
                if test_mAP>best_mAP:
                    best_mAP=test_mAP
                    utils.save_best_record(test_info,os.path.join(args.output_path, "best_record_seed_{}.txt".format(args.seed)),args.dataset)

                    torch.save(net.state_dict(), os.path.join(args.model_path,"model_seed_{}.pkl".format(args.seed)))
                print("\n Current test_mAP:{:.4f} best_mAP:{:.4f}".format(test_mAP,best_mAP))

    t_end = time.perf_counter()
    t_mi=(t_end-t_start)/60
    print(':Running time: %s minutes'%(t_mi))
                 
        

