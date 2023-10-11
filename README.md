<h1 align="center">A Demo for Temporal Action Localization</h1>

<img alt="Static Badge" src="https://img.shields.io/badge/Gradio-3.47.1-orange"><img alt="Static Badge" src="https://img.shields.io/badge/Temporal_Action_Localization-blue"><img alt="GitHub last commit (by committer)" src="https://img.shields.io/github/last-commit/pipixin321/TAL_APP">





### This is a Temporal Action Localization Demo based on the  [`gradio`](https://www.gradio.app/), **Temporal Action Localization** attempts to temporally **localize** and **classify** action instances in the untrimmed video, you can reference this repo if you want to build app for other video understanding tasks.

## 🐍:Demo Overview
<p>
  <div align=center><img src="./figs/demo.png" width="800" /></div>
</p>

## 📖:Installation
### 1. create conda environment & install pytorch
```bash
conda create --name tal_app python=3.8
conda activate tal_app
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. install mmaction2
- use MIM install MMEngine, MMCV.
```bash
pip install -U openmim
mim install mmengine
<!-- mim install mmcv -->
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
```

- install MMAction2.
```bash
cd Video-Swin-Transformer
pip install -v -e .
cd ..
```

### 3. install denseflow (option*)(install if optical flow is needed)
https://github.com/open-mmlab/denseflow/blob/master/INSTALL.md


### 4. compilation of nms
Part of NMS is implemented in C++. The code can be compiled by
```shell
cd ./tal_alg/actionformer/libs/utils
python setup.py install --user
```

### 5. install additional packages
```shell
cd ../../../../../
pip install -r requirements.txt
```


  
## 📑checkpoint&examples.
Download examples and checkpoint
- use `wget` to download the backbone checkpoint listed in ./backbone/download.py
- we provide checkpoint of **ActionFormer** trained on thumos14 dataset and testing examples, download them and put them in `./tal_alg/actionformer/ckpt` and `./examples`, respectively.
- [Download Link](https://pan.baidu.com/s/19yKxc13dIHcsyw5JtDZbkg?pwd=mdwj)

## 🚗Training(option*)
If you want to train  a customised model, following the steps below.
- Extract RGB and Flow Frame of video
RGB:
```bash
python extract_rawframes.py ./tmp/video/ ./tmp/rawframes/ --level 1 --ext mp4 --task rgb --use-opencv
```

- Extract features of dataset
Some videos are too long and cannot be loaded into memory when running in parallel. 
Filtering out the overly-long videos by param 'max-frame', the overly-long videos will be divided to <max-frame> picies.
```bash
cd dataset
python extract_datasets_feat.py --gpu-id <gpu> --part <part> --total <total>  --resume --max-frame 10000
```

- Train your temporal action localization algorithm
- Write a `inference.py` and import it in `processor.py`
  

## ✈️Run Demo
- set demo.launch(share=True) if you want to share your app to others.
- The whole process runs on the host server so the client(PC,Android,apple...) does not need to install the environment.
```bash
python main.py
```

## ❔Note
- 若未生成外部访问网站, 将frpc_linux_amd64_v0.2置于anaconda3/envs/tal_app/lib/python3.8/site-packages/gradio中
- 若未安装ffmpeg
```bash
sudo apt-get install ffmpeg
```

## 📝References
We referenced the repos below for the code.

* [VideoSwinTransfomer](https://github.com/MohammadRezaQaderi/Video-Swin-Transformer)
* [ActionFormer](https://github.com/happyharrycn/actionformer_release)
