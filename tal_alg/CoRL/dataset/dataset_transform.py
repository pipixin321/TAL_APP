import numpy as np
import os
from tqdm import tqdm

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

dataset='GTEA'
data_root='/root/Weakly_TAL/baseline_v1/dataset/{}'.format(dataset)


data=np.load(os.path.join(data_root,'{}-I3D-JOINTFeatures.npy'.format(dataset)),allow_pickle=True)

classlist=np.load(os.path.join(data_root,'classlist.npy'),allow_pickle=True)
subset=np.load(os.path.join(data_root,'subset.npy'),allow_pickle=True)
duration=np.load(os.path.join(data_root,'duration.npy'),allow_pickle=True)
videoname=np.load(os.path.join(data_root,'videoname.npy'),allow_pickle=True)

segments=np.load(os.path.join(data_root,'segments.npy'),allow_pickle=True)
labels=np.load(os.path.join(data_root,'labels.npy'),allow_pickle=True)


#class_list
classlist=[c.decode('utf-8') for c in classlist]
print(classlist)

#split train/test
train_data_dir=os.path.join(data_root,'train')
makedir(train_data_dir)
test_data_dir=os.path.join(data_root,'test')
makedir(test_data_dir)


split_train_dir=os.path.join(data_root,'split_train.txt')
split_test_dir=os.path.join(data_root,'split_test.txt')
split_train_lst,split_test_lst=[],[]
for i in tqdm(range(data.shape[0]),total=data.shape[0]):
    _data=data[i]
    _videoname=videoname[i].decode('utf-8')
    _subset=subset[i].decode('utf-8')
    if _subset=='training':
        split_train_lst.append(_videoname)
        np.save(os.path.join(train_data_dir,_videoname+'.npy'),_data)
    elif _subset=='validation':
        split_test_lst.append(_videoname)
        np.save(os.path.join(test_data_dir,_videoname+'.npy'),_data)

split_train_lst = [i+'\n' for i in split_train_lst]
with open(split_train_dir,'w') as f:
    f.writelines(split_train_lst)

split_test_lst = [i+'\n' for i in split_test_lst]
with open(split_test_dir,'w') as f:
    f.writelines(split_test_lst)  