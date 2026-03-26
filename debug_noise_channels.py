"""
Debug script: saves the 6 individual channels of y_cond (noisy input) and
y_noisy (Gaussian-noised clean image) to inspect whether the diffusion
noise is truly grayscale.
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from data.dataset import DenoisingDataset
import os

OUT_DIR = "debug_channels"
os.makedirs(OUT_DIR, exist_ok=True)

# Load one sample using the same dataset config as training
ds = DenoisingDataset(
    data_root="data",
    clean_dir="train_clean",
    noisy_dir="train_noisy",
    image_size=[256, 256],
)
sample = ds[0]
y_cond = sample['cond_image']   # noisy input,  shape [3, H, W], range [-1, 1]
y_0    = sample['gt_image']     # clean target,  shape [3, H, W], range [-1, 1]

# Generate Gaussian noise exactly as network.py:forward does (line 115)
noise_1ch = torch.randn(1, 1, y_0.shape[1], y_0.shape[2])
noise = noise_1ch.repeat(1, 3, 1, 1)  # [1, 3, H, W]

# Pick a random gamma value (mimicking a random timestep)
# Using gamma=0.5 so both signal and noise are clearly visible
sample_gamma = torch.tensor([[[[0.5]]]])
y_noisy = sample_gamma.sqrt() * y_0.unsqueeze(0) + (1 - sample_gamma).sqrt() * noise  # [1, 3, H, W]
y_noisy = y_noisy.squeeze(0)  # [3, H, W]


def save_channel(tensor, name):
    """Save a single-channel tensor (range [-1,1]) as a grayscale PNG."""
    arr = ((tensor.clamp(-1, 1) + 1) / 2 * 255).numpy().astype(np.uint8)
    Image.fromarray(arr, mode='L').save(os.path.join(OUT_DIR, f"{name}.png"))
    print(f"Saved {name}.png  min={tensor.min():.3f} max={tensor.max():.3f}")


# Save y_cond channels (the noisy input image from your dataset)
for ch, color in enumerate(["R", "G", "B"]):
    save_channel(y_cond[ch], f"y_cond_ch{ch}_{color}")

# Save y_noisy channels (clean image + diffusion Gaussian noise)
for ch, color in enumerate(["R", "G", "B"]):
    save_channel(y_noisy[ch], f"y_noisy_ch{ch}_{color}")

# Also save the noise itself per channel so you can compare directly
for ch, color in enumerate(["R", "G", "B"]):
    save_channel(noise[0, ch], f"noise_ch{ch}_{color}")

print(f"\nAll images saved to {OUT_DIR}/")
print(f"If noise is grayscale, the 3 noise channel images should look identical.")
print(f"Similarly, the 3 y_noisy channels should differ only by the underlying signal, not by noise pattern.")
