## 安装
首先介绍安装 SyncVSR 包的方案，具体的步骤如下（建议使用 python==3.9）
```
# 安装必要的系统包
apt-get update
apt-get -yq install ffmpeg libsm6 libxext6
apt install libturbojpeg tmux -y

# 安装合适版本的 PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其它包（使用清华源加速）
pip install pytorch-lightning -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==1.5.3 numpy==1.26.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install fairseq omegaconf sentencepiece transformers scikit-learn pydub timm wandb -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -U git+https://github.com/lilohuang/PyTurboJPEG.git --verbose
pip install opencv-python ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

接下来下载预训练的模型，可以通过代理进行加速：
```
# 下载 SyncVSR 的预训练模型
wget https://www.ghproxy.cn/https://github.com/KAIST-AILab/SyncVSR/releases/download/weight-audio-v1/Vox+LRS2+LRS3.ckpt
# 下载人脸检测 yolo-v8 的官方预训练模型
wget https://www.ghproxy.cn/https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt
```

## 使用 SyncVSR 对视频流进行 embedding
```
from SyncVSR import create_model
import torch

model = create_model(pretrained_path='./Vox+LRS2+LRS3.ckpt')
input_video = torch.randn((64, 1, 96, 96))  # 64 frames of 1-channel 96*96 video.
print(model(input_video))  # Decoded string
print(model.encode(input_video))  # Encoded video sequence: [64, 768]
```

## 完整的 demonstration
请执行 ``demo/demo_vsr.py``，加载一个视频片段，用来完整地进行一次 VSR 实验。

