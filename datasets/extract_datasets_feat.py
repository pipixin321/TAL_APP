import sys 
sys.path.append("..")
from extract_features import set_model,extract_feat
import os
from backbone.model_cfgs import MODEL_CFGS

model_name='swin_base'
data_pipeline,model=set_model(MODEL_CFGS[model_name])


# src='/mnt/data1/dataset/THUMOS14/frames/val_frames'
src='/mnt/data1/dataset/THUMOS14/frames/test_frames'

out='/mnt/data1/zhx/TAL_APP/datasets/THUMOS14/features/{}'.format(model_name)
if not os.path.exists(out):
    os.makedirs(out)
FEAT_CFGS={
        'data_prefix':src,
        'output_prefix':out,
        'batch_size':8,
        'modality':'RGB',
    }
extract_feat(data_pipeline,model,FEAT_CFGS)
print('Finish extract feature')

