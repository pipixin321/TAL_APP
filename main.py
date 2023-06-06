import gradio as gr
import os
import cv2
import shutil
from processor import process_video

current_dir = os.path.dirname(os.path.abspath(__file__))
def tal_func(video,progress=gr.Progress(track_tqdm=True)):
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
    results=process_video(tmp_dir)

    outvid='./tmp/result.mp4'
    return outvid,results


inputs=gr.Video()
outputs=['playable_video',gr.JSON()]
demo = gr.Interface(tal_func, 
                    inputs, 
                    outputs, 
                    examples=[os.path.join(os.path.dirname(__file__), 
                                "examples/video_test_0000004.mp4")],
                    cache_examples=False)

if __name__ == "__main__":
    # demo.launch(share=True,auth=('zhx','123'))
    # demo.launch(share=False)
    demo.queue(concurrency_count=20,status_update_rate=0.1).launch(share=False)



