import sys 
sys.path.append("..")
from extract_features import set_model,extract_feat
import os
from backbone.model_cfgs import MODEL_CFGS

model_name='csn' #['swin_tiny', 'slowfast101', 'csn', 'i3d']
data_pipeline,model=set_model(MODEL_CFGS[model_name])


gpu = 3
src='/home/dancer/datasets/TAL/thumos14/frames/train'
out='/home/dancer/datasets/TAL/thumos14/features/{}/rgb_train'.format(model_name)

# src='/home/dancer/datasets/TAL/thumos14/frames/test'
# out='/home/dancer/datasets/TAL/thumos14/features/{}/rgb_test'.format(model_name)


if not os.path.exists(out):
    os.makedirs(out)
FEAT_CFGS={
        'data_prefix':src,
        'output_prefix':out,
        'batch_size':8,
        'modality':'RGB',
        'backbone':model_name,
    }
extract_feat(data_pipeline,model,FEAT_CFGS, gpu)
print('Finish extract feature')

