"""
Debug script: loads trained model, runs the reverse diffusion process on one
image, and plots y_t at multiple timesteps to visualize where color artifacts appear.
Also plots the UNet's predicted noise channels at each sampled timestep.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from data.dataset import DenoisingDataset
from models.network import Network, make_beta_schedule, extract

CHECKPOINT = "experiments/train_denoising_experiment_260317_205551/checkpoint/best_epoch30_Network.pth"

# Build network with same config as training
net = Network(
    unet={
        "in_channel": 6,
        "out_channel": 3,
        "inner_channel": 64,
        "channel_mults": [1, 2, 4, 8],
        "attn_res": [16],
        "num_head_channels": 32,
        "res_blocks": 2,
        "dropout": 0.2,
        "image_size": 256,
    },
    beta_schedule={
        "train": {
            "schedule": "linear",
            "n_timestep": 1000,
            "linear_start": 1e-6,
            "linear_end": 0.01,
        },
        "test": {
            "schedule": "linear",
            "n_timestep": 1000,
            "linear_start": 1e-6,
            "linear_end": 0.01,
        },
    },
    module_name="guided_diffusion",
    init_type="kaiming",
)

# Load weights
state_dict = torch.load(CHECKPOINT, map_location="cpu")
net.load_state_dict(state_dict, strict=False)
net.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
net.set_new_noise_schedule(device=device, phase="test")

# Load one sample
ds = DenoisingDataset(
    data_root="data",
    clean_dir="train_clean",
    noisy_dir="train_noisy",
    image_size=[256, 256],
)
sample = ds[0]
y_cond = sample["cond_image"].unsqueeze(0).to(device)  # [1, 3, H, W]
y_0 = sample["gt_image"].unsqueeze(0).to(device)

# Start from pure grayscale noise (same as restoration does)
y_t = torch.randn(1, 1, 256, 256, device=device).repeat(1, 3, 1, 1)

num_timesteps = net.num_timesteps
# Save snapshots at these timesteps (evenly spaced + final)
snapshot_steps = list(range(num_timesteps - 1, -1, -(num_timesteps // 8)))
if 0 not in snapshot_steps:
    snapshot_steps.append(0)
snapshot_steps = sorted(snapshot_steps, reverse=True)

snapshots_yt = {}
snapshots_pred_noise = {}

print(f"Running reverse process for {num_timesteps} timesteps...")
print(f"Saving snapshots at t = {snapshot_steps}")

with torch.no_grad():
    for i in range(num_timesteps - 1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)

        # Get model prediction (predicted noise) before p_sample
        if i in snapshot_steps:
            noise_level = extract(net.gammas, t, x_shape=(1, 1)).to(device)
            pred_noise = net.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
            snapshots_pred_noise[i] = pred_noise.cpu().squeeze(0)
            snapshots_yt[i] = y_t.cpu().squeeze(0).clone()

        # Actual p_sample step
        y_t = net.p_sample(y_t, t, y_cond=y_cond)

    # Save final output
    snapshots_yt[-1] = y_t.cpu().squeeze(0).clone()

# --- Plot y_t at each snapshot ---
all_steps = sorted(snapshots_yt.keys(), reverse=True)
n = len(all_steps)

fig, axes = plt.subplots(3, n, figsize=(3 * n, 9))
fig.suptitle("y_t per channel at different timesteps (R / G / B rows)", fontsize=14)

for col, t_step in enumerate(all_steps):
    img = snapshots_yt[t_step]  # [3, H, W]
    label = f"t={t_step}" if t_step >= 0 else "final"
    for ch in range(3):
        channel_data = ((img[ch].clamp(-1, 1) + 1) / 2).numpy()
        axes[ch, col].imshow(channel_data, cmap="gray", vmin=0, vmax=1)
        axes[ch, col].set_title(label if ch == 0 else "")
        axes[ch, col].axis("off")
    axes[0, col].set_title(label)

axes[0, 0].set_ylabel("R", fontsize=12)
axes[1, 0].set_ylabel("G", fontsize=12)
axes[2, 0].set_ylabel("B", fontsize=12)

plt.tight_layout()
plt.savefig("debug_channels/reverse_yt_channels.png", dpi=150)
print("Saved debug_channels/reverse_yt_channels.png")

# --- Plot predicted noise at each snapshot ---
noise_steps = sorted(snapshots_pred_noise.keys(), reverse=True)
n2 = len(noise_steps)

fig2, axes2 = plt.subplots(3, n2, figsize=(3 * n2, 9))
fig2.suptitle("Predicted noise per channel at different timesteps (R / G / B rows)", fontsize=14)

for col, t_step in enumerate(noise_steps):
    noise = snapshots_pred_noise[t_step]  # [3, H, W]
    for ch in range(3):
        channel_data = noise[ch].numpy()
        axes2[ch, col].imshow(channel_data, cmap="gray")
        axes2[ch, col].axis("off")
    axes2[0, col].set_title(f"t={t_step}")

axes2[0, 0].set_ylabel("R", fontsize=12)
axes2[1, 0].set_ylabel("G", fontsize=12)
axes2[2, 0].set_ylabel("B", fontsize=12)

plt.tight_layout()
plt.savefig("debug_channels/reverse_pred_noise_channels.png", dpi=150)
print("Saved debug_channels/reverse_pred_noise_channels.png")

# --- Also plot y_t as RGB to see color artifacts ---
fig3, axes3 = plt.subplots(1, n, figsize=(3 * n, 3))
fig3.suptitle("y_t as RGB at different timesteps", fontsize=14)

for col, t_step in enumerate(all_steps):
    img = snapshots_yt[t_step]  # [3, H, W]
    rgb = ((img.clamp(-1, 1) + 1) / 2).permute(1, 2, 0).numpy()
    axes3[col].imshow(rgb)
    axes3[col].set_title(f"t={t_step}" if t_step >= 0 else "final")
    axes3[col].axis("off")

plt.tight_layout()
plt.savefig("debug_channels/reverse_yt_rgb.png", dpi=150)
print("Saved debug_channels/reverse_yt_rgb.png")

plt.show()
print("Done!")
