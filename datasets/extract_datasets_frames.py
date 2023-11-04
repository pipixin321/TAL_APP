import sys 
sys.path.append("..")
from extract_rawframes import extract_from_video


# vid_path = '/home/dancer/datasets/TAL/thumos14/videos/train'
# frame_path = '/home/dancer/datasets/TAL/thumos14/frames/train'
vid_path = '/home/dancer/datasets/TAL/thumos14/videos/test'
frame_path = '/home/dancer/datasets/TAL/thumos14/frames/test'
FRAME_CFGS={
                'src_dir':vid_path,
                'out_dir':frame_path,
                'task':'rgb',
                'level':1,
                'ext':'mp4',
                'new_short':180,
                'use_opencv':True
            }
extract_from_video(FRAME_CFGS)
print('Finish extract raw frames')

