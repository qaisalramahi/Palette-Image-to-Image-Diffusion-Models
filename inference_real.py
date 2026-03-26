"""
Run the trained 100-step denoising model on real sonar images (no GT needed).
Saves denoised outputs and a side-by-side comparison grid.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from models.network import Network, extract
from pathlib import Path
import os

CHECKPOINT = "experiments/y_t_cond_image_100_n_timesteps/checkpoint/best_epoch90_Network.pth"
INPUT_DIR = "real_new_blue"
OUTPUT_DIR = "real_new_blue_denoised"
IMAGE_SIZE = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build network
net = Network(
    unet={
        "in_channel": 2,
        "out_channel": 1,
        "inner_channel": 64,
        "channel_mults": [1, 2, 4, 8],
        "attn_res": [16],
        "num_head_channels": 32,
        "res_blocks": 2,
        "dropout": 0.2,
        "image_size": IMAGE_SIZE,
    },
    beta_schedule={
        "train": {
            "schedule": "linear",
            "n_timestep": 100,
            "linear_start": 1e-06,
            "linear_end": 0.01,
        },
        "test": {
            "schedule": "linear",
            "n_timestep": 100,
            "linear_start": 1e-06,
            "linear_end": 0.01,
        },
    },
    module_name="guided_diffusion",
    init_type="kaiming",
)

state_dict = torch.load(CHECKPOINT, map_location="cpu")
net.load_state_dict(state_dict, strict=False)
net.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
net.set_new_noise_schedule(device=device, phase="test")

# Preprocessing (same as DenoisingDataset)
tfs = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Collect input images
input_paths = sorted(Path(INPUT_DIR).glob("*.jpg")) + sorted(Path(INPUT_DIR).glob("*.png"))
print(f"Found {len(input_paths)} images in {INPUT_DIR}")

results = []

with torch.no_grad():
    for img_path in input_paths:
        print(f"Processing {img_path.name}...")

        # Load as grayscale and preprocess
        img_pil = Image.open(img_path).convert("L")
        y_cond = tfs(img_pil).unsqueeze(0).to(device)  # [1, 1, 256, 256]

        # Run restoration starting from y_cond
        output, visuals = net.restoration(y_cond, y_t=y_cond, sample_num=1)

        # Convert to [0, 1] for saving
        input_np = ((y_cond.cpu().squeeze().clamp(-1, 1) + 1) / 2).numpy()
        output_np = ((output.cpu().squeeze().clamp(-1, 1) + 1) / 2).numpy()

        # Save individual denoised image
        out_pil = Image.fromarray((output_np * 255).astype(np.uint8), mode="L")
        out_pil.save(os.path.join(OUTPUT_DIR, img_path.name))

        results.append((img_path.name, input_np, output_np))

print(f"\nSaved {len(results)} denoised images to {OUTPUT_DIR}/")

# --- Grid comparison ---
n = len(results)
cols = min(6, n)
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows * 2, cols, figsize=(3 * cols, 3 * rows * 2))
if rows * 2 == 1:
    axes = axes[np.newaxis, :]
if cols == 1:
    axes = axes[:, np.newaxis]

fig.suptitle("Input (top) vs Denoised (bottom)", fontsize=14)

for idx, (name, inp, out) in enumerate(results):
    r = (idx // cols) * 2
    c = idx % cols

    axes[r, c].imshow(inp, cmap="gray", vmin=0, vmax=1)
    axes[r, c].set_title(name, fontsize=7)
    axes[r, c].axis("off")

    axes[r + 1, c].imshow(out, cmap="gray", vmin=0, vmax=1)
    axes[r + 1, c].axis("off")

# Hide unused axes
for idx in range(n, rows * cols):
    r = (idx // cols) * 2
    c = idx % cols
    axes[r, c].axis("off")
    axes[r + 1, c].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_grid.png"), dpi=150)
print(f"Saved {OUTPUT_DIR}/comparison_grid.png")
plt.show()
print("Done!")
