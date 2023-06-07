# A Demo for Temporal Action Localization
<p>
  <img src="./figs/demo.png" width="800" />
</p>

## 1.Enviroment
## 1.1 create conda environment & install pytorch
```bash
conda create --name tal_app python=3.7
conda activate tal_app
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 1.2 install mmaction2
1.install MMEngine, MMCV, using MIM.
```bash
pip install -U openmim
mim install mmengine
<!-- mim install mmcv -->
pip install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
```

2.Install MMAction2.
Install from source(Recommended):
<!-- git clone https://github.com/open-mmlab/mmaction2.git -->
```bash
git clone git@github.com:MohammadRezaQaderi/Video-Swin-Transformer.git
cd mmaction2
pip install -v -e .
```

- Verify the installation of mmaction2
```bash
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```
~~~~
output:
arm wrestling:  1.0
rock scissors paper:  9.462872515740225e-14
massaging feet:  1.6511704519937484e-14
stretching leg:  2.960374221423149e-15
opening bottle:  2.2746777572147433e-15
~~~~

## 1.3 install denseflow (option*)(install if optical flow is needed)
https://github.com/open-mmlab/denseflow/blob/master/INSTALL.md


## 2.Workflow
- Extract RGB and Flow
RGB:
```bash
python extract_rawframes.py ./tmp/video/ ./tmp/rawframes/ --level 1 --ext mp4 --task rgb --use-opencv
```

- Extract features
Some videos are too long and cannot be loaded into memory when running in parallel. 
Filtering out the overly-long videos by param 'max-frame',the overly-long videos will be divided to <max-frame> picies.
```bash
cd dataset
python extract_datasets_feat.py --part <part> --total <total> --resume --max-frame 15000
```

## 3.Demo built with gradio
- set demo.launch(share=True) if you want to share your app to others.
- The whole process runs on the host server so the client(PC,Android,apple...) does not need to install the environment.
```bash
python main.py
```
