import os
import shutil
import gradio as gr
from processor import process_video

current_dir = os.path.dirname(os.path.abspath(__file__))

def tal_func(video, new_short, backbone, detector, score_thresh, overlap_thresh,
             progress=gr.Progress(track_tqdm=True)):
    print(video)
    tmp_dir=os.path.join(current_dir,'tmp')
    vid_path=os.path.join(tmp_dir,'video')

    vid_name=video.split('/')[-1]
    if not os.path.exists(os.path.join(vid_path, vid_name)):
        print('Creating tmp folder')
        if os.path.exists(tmp_dir): 
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
        os.makedirs(vid_path)
        print('Copying file...')
        new_video=os.path.join(vid_path,vid_name)
        shutil.copyfile(video,new_video)
        # os.remove(video)

    print('Processing video...')
    postprocess_cfgs = {'score_thresh':score_thresh, 'overlap_thresh':overlap_thresh}
    results=process_video(tmp_dir, vid_name, new_short, backbone, detector, postprocess_cfgs)

    outvid='./tmp/result.mp4'
    return outvid,results


if __name__ == "__main__":
    inputs=[gr.Video(label='输入视频 (Input Video)'),
        gr.Slider(0, 640, value=180, step=10, label="视频尺寸 (video size)", info="调整输入视频短边长度, 0为原始尺寸(resize video's short side to a new value, set 0 to keep the original size)"),
        gr.inputs.Radio(['I3D','SlowFast','CSN','SwinViViT'], default='I3D', label='视频特征提取网络 (Video Feature Extraction Network)'),
        gr.inputs.Radio(['ActionFormer(Fully-supervised)','CoRL(Weakly-Supervised)'], default='ActionFormer(Fully-supervised)', label='时序行为检测网络 (Temporal Action Localization Network)'),
        gr.Slider(0, 1, value=0.2, step=0.1, label='[后处理]置信度阈值 ([Postprocess]Score threshold)'),
        gr.Slider(0, 1, value=0.9, step=0.1, label='[后处理]重叠阈值 ([Postprocess]Overlap threshold)')]
    outputs=[gr.Video(label='输出视频 (Output Video)'), gr.JSON(label='检测结果 (Localization Results)')] #'playable_video'
    examples=[
        ["./examples/video_test_0000062.mp4",180,'SwinViViT','ActionFormer(Fully-supervised)'],
        ["./examples/video_test_0000635.mp4",180,'I3D','CoRL(Weakly-Supervised)'],
        ["./examples/video_test_0000450.mp4",180,'SwinViViT','ActionFormer(Fully-supervised)'],
    ]
    demo = gr.Interface(tal_func, 
                        inputs, 
                        outputs, 
                        examples=examples,
                        cache_examples=False,
                        theme='soft',  #options: base, Monochrome, Soft, Glass
                        title='时序行为检测演示界面 (GUI Demo Of Temporal Action Localization)',
                        description='时序行为检测是指在时序上定位未裁剪长视频中的行为片段并分类 \
                                    \n(Temporal Action Localization attempts to temporally localize and classify action instances in the untrimmed video) \
                                    \n输入:视频(.mp4) 输出:检测后的视频(.mp4) & 检测结果 \
                                    \n(Input:Video(.mp4) Output:Video(.mp4) & Localization Results)')

    
    # demo.launch(share=True,auth=('zhx','123'))
    # demo.launch(share=False)
    demo.queue(concurrency_count=1,max_size=1).launch(share=False)



