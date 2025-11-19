import sys
import cv2
sys.path.insert(0, ".")
from seedvr.pipeline import SeedVRPipeline
from PIL import Image
from einops import rearrange
import torch
import numpy as np
import datetime
from cv2 import VideoCapture
import mediapy

from seedvr.common.distributed import (
    init_torch,
    get_world_size,
    get_device,
)
from seedvr.common.distributed.advanced import (
    init_sequence_parallel,
)

if get_world_size() > 1:
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    init_sequence_parallel(get_world_size())

#pipeline = SeedVRPipeline.from_original_pretrained(device="cuda")
#pipeline.dit.to(torch.bfloat16)
#pipeline.save_pretrained_flashpack("SeedVR2-7B-FlashPack")
pipeline = SeedVRPipeline.from_pretrained_flashpack("fal/SeedVR2-7B-FlashPack", device=get_device(), use_distributed_loading=get_world_size() > 1)
"""
samples = pipeline(
    image,
    height=2768,
    width=2144,
)
samples = [
    sample[0].permute(1, 2, 0)
    for sample in samples
]
samples = [
    Image.fromarray(sample.numpy())
    for sample in samples
]  # convert to PIL Image
for i, sample in enumerate(samples):
    sample.save(f"sample_{i}.png")

"""

# Now test video
max_frames = 42
video_array = []
video_capture = VideoCapture("test_video.mp4")
i = 0
while i < max_frames:
    ret, frame = video_capture.read()
    if not ret:
        break
    i += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_array.append(frame)
video_capture.release()

video_array = np.array(video_array)
video_array = torch.from_numpy(video_array)
video_array = video_array.permute(0, 3, 1, 2)
video_array = video_array.to(torch.float32) / 255.0
video_array = video_array * 2.0 - 1.0
video_array = rearrange(video_array, "t c h w -> c t h w")

c, t, h, w = video_array.shape

samples = pipeline(
    media=video_array,
    target_area=1080*1920,
    batch_size=33,
    temporal_overlap=30,
    use_tiling=True,
)
print(samples.shape)
# c t h w -> t h w c
samples = samples.permute(1, 2, 3, 0)
print(samples.shape)
mediapy.write_video("test_video_output.mp4", samples.numpy().astype(np.uint8), fps=30)