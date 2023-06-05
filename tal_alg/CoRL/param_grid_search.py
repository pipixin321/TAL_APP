import sys 
import os
import json
from options import parse_args
import utils
import torch
from model import Model,PGM
from test import test
import torch.utils.data as data
from dataset import dataset
from tensorboard_logger import Logger
import numpy as np

import matplotlib.pyplot as plt

plt.switch_backend('agg')

def after_processing_param():
    save_path='/mnt/data1/zhx/TAL_APP/tal_alg/CoRL/grid_search'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    args=parse_args()
    utils.set_seed(args.seed)
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    args.device = device
    net=Model(args)
    net=net.to(device)
    test_loader=data.DataLoader(dataset(args,phase="test",sample="random",stage=2),
                                    batch_size=1,shuffle=False, num_workers=args.num_workers)
    test_info = args.test_info
    logger=Logger(args.log_path)
    if args.checkpoint=='':
        args.checkpoint=os.path.join(args.model_path,"model_seed_{}.pkl".format(args.seed))
    net.load_state_dict(torch.load(args.checkpoint))

    pgnet=PGM(args)
    pgnet=pgnet.to(args.device)


    print("Test Start")
    result_dict={}
    ##############################
    param='nms_thresh'
    param_range=np.arange(0.3,0.9,0.1)
    # param_range = [item.strip() for item in list(open('./grid_search/bad_case_test.txt'))]
    # initial_blocked_videos=args.blocked_videos+param_range
    ##############################
    result_dict['param']=param
    result_dict['result']={}
    file_path=os.path.join(save_path,'param_{}_results.json'.format(result_dict['param']))
    if os.path.exists(file_path):
        with open(file_path,'r') as json_file:
            result_dict=json.load(json_file)
            print("load from {}".format(file_path))
    for para in param_range:
        para=round(para,3)
        ########################
        args.nms_thresh=para
        # args.act_thresh_cas=np.arange(0, para, 0.025)
        # args.blocked_videos=initial_blocked_videos

        mAP=test(net,pgnet,args,test_loader,logger,0,test_info,model_file=None,save_file=False,subset='test')
        print(param,para,mAP)
        result_dict['result'][str(para)]=mAP
        save_dict=json.dumps(result_dict,indent=4)
        with open(file_path,'w') as json_file:
            json_file.write(save_dict)
    plot_param_curve(param=param)

def plot_param_curve(param='nms_thresh'):
    
    save_path='./grid_search'
    file_path=os.path.join(save_path,'param_{}_results.json'.format(param))
    if os.path.exists(file_path):
        with open(file_path,'r') as json_file:
            result_dict=json.load(json_file)
            print("load from {}".format(file_path))
    result=result_dict['result']

    plt.figure()
    for k,v in result.items():
        plt.plot(float(k),v,marker="o",markersize=4,color='red')
    plt.grid(True)
    plt.savefig(os.path.join(save_path,'curve_{}.png'.format(param)),dpi=300)

after_processing_param()
# plot_param_curve(param='gamma')