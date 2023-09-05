import gradio as gr
import os
import cv2
import shutil
from processor import process_video

current_dir = os.path.dirname(os.path.abspath(__file__))
def tal_func(video,new_short,backbone,detector,
             progress=gr.Progress(track_tqdm=True)):
    print(video)
    print('Creating tmp folder')
    tmp_dir=os.path.join(current_dir,'tmp')
    if os.path.exists(tmp_dir): 
        shutil.rmtree(tmp_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    vid_path=os.path.join(tmp_dir,'video')
    if not os.path.exists(vid_path):
        os.makedirs(vid_path)
    print('Video loaded {}'.format(video))
    vid_name=video.split('/')[-1]
    print('Copying file...')
    new_video=os.path.join(vid_path,vid_name)
    shutil.copyfile(video,new_video)
    os.remove(video)

    print('Processing video...')
    results=process_video(tmp_dir,new_short,backbone,detector)

    outvid='./tmp/result.mp4'
    return outvid,results


inputs=[gr.Video(label='Input Video'),
        gr.Slider(0, 640, value=180, step=10, label="video size", info="resize video's short side to a new value,set 0 to keep the original size"),
        gr.inputs.Radio(['I3D','SlowFast','CSN','SwinViViT'],default='SwinViViT',label='backbone'),
        gr.inputs.Radio(['ActionFormer(Fully-supervised)','CoRL(Weakly-Supervised)'],default='CoRL(Weakly-Supervised)',label='detector'),]
outputs=[gr.Video(label='Output Video'), gr.JSON(label='Detection Results')] #'playable_video'
examples=[
    ["./examples/video_test_0000062.mp4",180,'SwinViViT','ActionFormer(Fully-supervised)'],
    ["./examples/video_test_0000635.mp4",180,'SwinViViT','ActionFormer(Fully-supervised)'],
    ["./examples/video_test_0000450.mp4",180,'SwinViViT','ActionFormer(Fully-supervised)'],
]
demo = gr.Interface(tal_func, 
                    inputs, 
                    outputs, 
                    examples=examples,
                    cache_examples=False,
                    title='Demo For Temporal Action Localization',
                    description='Temporal Action Localization(TAL) attempts to temporally localize and classify \
                                    action instances in the untrimmed video. \
                                    \nInput:Video(.mp4) \nOutput:Localization Results')

if __name__ == "__main__":
    # demo.launch(share=True,auth=('zhx','123'))
    # demo.launch(share=False)
    demo.queue(concurrency_count=1,max_size=1).launch(share=True)



