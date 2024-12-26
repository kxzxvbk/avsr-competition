import torch

from SyncVSR import create_model
from SyncVSR.preprocess.prepare_LRS2 import extract_yolov8


if __name__ == "__main__":
    model = create_model().cuda()
    video = torch.tensor(extract_yolov8('demo/example.mp4'))
    print(f"The video tensor's shape is: {video.shape}")
    print(f"VSR result: {model(video)}")
    print(f"Embedding shape: {model.encode(video)}")
