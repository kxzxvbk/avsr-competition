import torch
import numpy as np
from turbojpeg import TJPF_GRAY, TurboJPEG

from SyncVSR import create_model
from SyncVSR.preprocess.prepare_LRS2 import extract_yolov8
jpeg = TurboJPEG()


def load_and_preprocess_data(video_path):
    total_video = extract_yolov8(video_path)
    total_video = np.stack([jpeg.decode(img, TJPF_GRAY) for img in total_video])
    total_video = torch.as_tensor(total_video).permute(0, 3, 1, 2)  # T x C x H x W
    return total_video


if __name__ == "__main__":
    model = create_model().cuda()
    video = load_and_preprocess_data(video_path='demo/example.mp4')
    print(f"The video tensor's shape is: {video.shape}")
    print(f"VSR result: {model(video)}")
    print(f"Embedding shape: {model.encode(video)}")
