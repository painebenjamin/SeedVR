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

#pipeline = SeedVRPipeline.from_original_pretrained(device="cuda")
#pipeline.save_pretrained_flashpack("SeedVR2-7B-FlashPack")
pipeline = SeedVRPipeline.from_pretrained_flashpack("./SeedVR2-7B-FlashPack", device="cuda")
samples = pipeline(
    image,
    height=1920,
    width=1080,
)
print(f"{samples[0].shape=} {samples[0].dtype=} {samples[0].min()} {samples[0].max()}")
samples = [Image.fromarray(sample.cpu().numpy()) for sample in samples]  # convert to PIL Image

for i, sample in enumerate(samples):
    sample.save(f"sample_{i}.png")