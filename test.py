import sys
sys.path.insert(0, ".")
from seedvr.pipeline import SeedVRPipeline
from PIL import Image
from einops import rearrange
import torch
import numpy as np

image = Image.open("test_image.jpg")
image = torch.from_numpy(np.array(image)) # hwc
image = rearrange(image, "h w c -> c 1 h w")  #bcfhw
image = image.to(torch.float32) / 255.0  # [0, 1]
image = image * 2.0 - 1.0  # normalize to [-1, 1]

pipeline = SeedVRPipeline.from_original_pretrained(device="cuda")
samples = pipeline(
    image,
    height=1280,
    width=720,
)
samples = [sample * 0.5 + 0.5 for sample in samples]  # normalize to [0, 1]
samples = [sample * 255.0 for sample in samples]  # denormalize to [0, 255]
samples = [sample.to(torch.uint8) for sample in samples]  # convert to uint8
samples = [sample.permute(1, 2, 0) for sample in samples]  # convert to CHW
print(f"{samples[0].shape=} {samples[0].dtype=} {samples[0].min()} {samples[0].max()}")
samples = [Image.fromarray(sample.cpu().numpy()) for sample in samples]  # convert to PIL Image

for i, sample in enumerate(samples):
    sample.save(f"sample_{i}.png")