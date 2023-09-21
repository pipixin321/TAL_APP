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
    trimmed_vid = './tmp/result_trimmed.mp4'
    return outvid, trimmed_vid, results

def train_network(filedir):
    # Todo: training function
    return filedir

if __name__ == "__main__":
    theme = 'soft' #options: base, Monochrome, Soft, Glass
    inputs=[gr.Video(label='输入视频 (Input Video)'),
        gr.Slider(0, 640, value=180, step=10, label="视频尺寸 (video size)", info="调整输入视频短边长度, 0为原始尺寸(resize video's short side to a new value, set 0 to keep the original size)"),
        gr.components.Radio(['I3D','SlowFast','CSN','SwinViViT'],  label='视频特征提取网络 (Video Feature Extraction Network)'),
        gr.components.Radio(['ActionFormer(Fully-supervised)','CoRL(Weakly-Supervised)'], label='时序行为检测网络 (Temporal Action Localization Network)'),
        gr.Slider(0, 1, value=0.2, step=0.1, label='[后处理]置信度阈值 ([Postprocess]Score threshold)'),
        gr.Slider(0, 1, value=0.9, step=0.1, label='[后处理]重叠阈值 ([Postprocess]Overlap threshold)')]
    outputs=[gr.Video(label='输出视频 (Output Video)'),
            gr.Video(label='裁剪后视频 (Trimmed Video)'),
            gr.JSON(label='检测结果 (Localization Results)')] #'playable_video'
    examples=[
        ["./examples/video_test_0001433.mp4", 180, 'I3D', 'ActionFormer(Fully-supervised)', 0.2, 0.9],
        ["./examples/video_test_0000062.mp4", 180, 'SwinViViT', 'ActionFormer(Fully-supervised)', 0.2, 0.9],
        ["./examples/video_test_0000635.mp4", 180, 'I3D', 'CoRL(Weakly-Supervised)', 0.2, 0.9],
        ["./examples/video_test_0000450.mp4", 180, 'SwinViViT', 'ActionFormer(Fully-supervised)', 0.2, 0.9],
    ]
    demo1 = gr.Interface(tal_func, 
                        inputs, 
                        outputs, 
                        examples=examples,
                        cache_examples=False,
                        theme=theme, 
                        # title='时序行为检测演示界面 (GUI Demo Of Temporal Action Localization)',
                        description="**时序行为检测**是指在时序上**定位**未裁剪长视频中的行为片段并**分类** \
                                    \n(**Temporal Action Localization** attempts to temporally **localize** and **classify** action instances in the untrimmed video) \
                                    \n- **输入**: 视频[.mp4] \
                                    \n- **输出**: 检测后的视频[.mp4] & 裁剪后的视频 & 检测结果")
    demo2 = gr.Interface(train_network, inputs=gr.File(label='数据集文件 (Dataset File)'), outputs=gr.Text(), theme=theme,
                        description='上传规范化的数据集压缩包,训练网络')
    
    css = ".gradio-container { \
    background-image: url('file=figs/HUST.jpg');\
    background-size: 100% 100%;\
    background-repeat: no-repeat;\
    background-attachment: fixed;}"
    demo = gr.TabbedInterface([demo1, demo2],
                              tab_names=['Localization', 'Train Network'],
                              title='时序行为检测演示界面 (WebUI Demo Of Temporal Action Localization)',
                              theme=theme,
                              css=css,
                              )
    
    # demo.launch(share=True,auth=('zhx','123'))
    demo.queue(concurrency_count=1,max_size=1).launch(share=True)



