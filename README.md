# SeedVR

SeedVR is a high-quality video and image super-resolution pipeline powered by diffusion models.

## Installation

```bash
pip install -e .
```

## Quick Start

### Image Super-Resolution

```python
import torch
import numpy as np
from PIL import Image
from einops import rearrange
from seedvr.pipeline import SeedVRPipeline

# Load and preprocess image
image = Image.open("test_2.png")
image = torch.from_numpy(np.array(image))  # HWC format
image = rearrange(image, "h w c -> c 1 h w")  # Convert to CFHW
image = image.to(torch.float32) / 255.0  # Normalize to [0, 1]
image = image * 2.0 - 1.0  # Normalize to [-1, 1]

# Load pipeline
pipeline = SeedVRPipeline.from_pretrained_flashpack(
    "fal/SeedVR2-7B-FlashPack",
    device="cuda"
)

# Generate super-resolution samples
samples = pipeline(
    image,
    target_area=850 * 850,
    use_tiling=False,
)

# Post-process and save
samples = [sample[0].permute(1, 2, 0) for sample in samples]
samples = [Image.fromarray(sample.numpy()) for sample in samples]

for i, sample in enumerate(samples):
    sample.save(f"sample_{i}.png")
```

### Video Super-Resolution

```python
import cv2
import torch
import numpy as np
import mediapy
from cv2 import VideoCapture
from einops import rearrange
from seedvr.pipeline import SeedVRPipeline

# Load video
video_array = []
video_capture = VideoCapture("test_video.mp4")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_array.append(frame)

video_capture.release()

# Preprocess video
video_array = np.array(video_array)
video_array = torch.from_numpy(video_array)
video_array = video_array.permute(0, 3, 1, 2)
video_array = video_array.to(torch.float32) / 255.0
video_array = video_array * 2.0 - 1.0
video_array = rearrange(video_array, "t c h w -> c t h w")

# Load pipeline
pipeline = SeedVRPipeline.from_pretrained_flashpack(
    "fal/SeedVR2-7B-FlashPack",
    device="cuda"
)

# Generate super-resolution video
samples = pipeline(
    video_array,
    target_area=2160 * 3840,
    batch_size=33,
    temporal_overlap=8,
)

# Save output
mediapy.write_video(
    "test_video_output.mp4",
    samples[0].permute(0, 2, 3, 1).numpy().astype(np.uint8),
    fps=30
)
```

## Distributed Inference

For multi-GPU setups:

```python
from seedvr.common.distributed import (
    init_torch,
    get_world_size,
    get_device,
)
from seedvr.common.distributed.advanced import init_sequence_parallel
import datetime

if get_world_size() > 1:
    init_torch(cudnn_benchmark=False, timeout=datetime.timedelta(seconds=3600))
    init_sequence_parallel(get_world_size())

pipeline = SeedVRPipeline.from_pretrained_flashpack(
    "fal/SeedVR2-7B-FlashPack",
    device=get_device(),
    use_distributed_loading=get_world_size() > 1
)
```

## Key Parameters

- `target_area`: Target resolution in pixels (e.g., `850 * 850` for images, `2160 * 3840` for 4K video)
- `use_tiling`: Enable tiling for large images (default: `False`)
- `batch_size`: Number of frames to process in each batch (video only)
- `temporal_overlap`: Number of overlapping frames between batches (video only)

## License

See [LICENSE](LICENSE) file for details.


