import torch
import torchvision
import numpy as np
from turbojpeg import TJPF_GRAY, TurboJPEG

from SyncVSR import create_model
from SyncVSR.preprocess.prepare_LRS2 import extract_yolov8
jpeg = TurboJPEG()


def load_and_preprocess_data(video_path, img_transform):
    # 对视频进行预处理，裁剪出 128x128的人脸区域，用于后续准确的唇语识别
    total_video = extract_yolov8(video_path)

    # 转换成目标格式
    total_video = np.stack([jpeg.decode(img, TJPF_GRAY) for img in total_video])
    total_video = torch.as_tensor(total_video).permute(0, 3, 1, 2) / 255.  # T x C x H x W
    return img_transform(total_video)


if __name__ == "__main__":
    model = create_model(pretrained_path='./Vox+LRS2+LRS3.ckpt').cuda().eval()

    # 对数据进行加载和预处理
    transform = torch.nn.Sequential(
        torchvision.transforms.CenterCrop(96),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Normalize(0.421, 0.165),
    )
    video = load_and_preprocess_data(video_path='demo/example.mp4', img_transform=transform)
    video = video[:32, ...]  # 节省内存，仅取前32帧来预测

    print(f"The video tensor's shape is: {video.shape}")
    print(f"VSR result: {model(video)}")
    print(f"Embedding shape: {model.encode(video).shape}")
