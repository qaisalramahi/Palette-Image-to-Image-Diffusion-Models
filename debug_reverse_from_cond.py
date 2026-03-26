"""
Debug script: visualize the reverse diffusion process when starting from y_t = y_cond
(the noisy sonar image) instead of pure Gaussian noise.

Shows y_t snapshots, predicted noise, and pixel-wise difference from GT at each step.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.dataset import DenoisingDataset
from models.network import Network, extract
import os

CHECKPOINT = "experiments/y_t_cond_image_100_n_timesteps/checkpoint/best_epoch90_Network.pth"

# Build network with same config as the 100-step training run
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
        "image_size": 256,
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

# Load weights
state_dict = torch.load(CHECKPOINT, map_location="cpu")
net.load_state_dict(state_dict, strict=False)
net.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)
net.set_new_noise_schedule(device=device, phase="test")

# Print schedule info
print(f"num_timesteps: {net.num_timesteps}")
print(f"gammas (cumulative alpha): min={net.gammas.min():.6f}, max={net.gammas.max():.6f}")
print(f"gamma[0]={net.gammas[0]:.6f}, gamma[49]={net.gammas[49]:.6f}, gamma[99]={net.gammas[99]:.6f}")

# Load one sample
ds = DenoisingDataset(
    data_root="data",
    clean_dir="test_clean",
    noisy_dir="test_noisy",
    image_size=[256, 256],
)
sample = ds[0]
y_cond = sample["cond_image"].unsqueeze(0).to(device)  # [1, 1, H, W]
gt = sample["gt_image"].unsqueeze(0).to(device)         # [1, 1, H, W]

print(f"y_cond range: [{y_cond.min():.3f}, {y_cond.max():.3f}]")
print(f"gt range: [{gt.min():.3f}, {gt.max():.3f}]")

# --- Run reverse process starting from y_cond ---
num_timesteps = net.num_timesteps
y_t = y_cond.clone()

# Save snapshots at evenly spaced steps + start/end
n_snapshots = 10
snapshot_steps = list(range(num_timesteps - 1, -1, -(num_timesteps // n_snapshots)))
if 0 not in snapshot_steps:
    snapshot_steps.append(0)
snapshot_steps = sorted(set(snapshot_steps), reverse=True)

snapshots_yt = {num_timesteps: y_t.cpu().squeeze(0).clone()}  # before any step
snapshots_pred_noise = {}
snapshots_pred_x0 = {}

print(f"\nRunning reverse process for {num_timesteps} timesteps, starting from y_cond...")
print(f"Saving snapshots at t = {snapshot_steps}")

with torch.no_grad():
    for i in range(num_timesteps - 1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)

        if i in snapshot_steps:
            # Capture predicted noise and predicted x0
            noise_level = extract(net.gammas, t, x_shape=(1, 1)).to(device)
            pred_noise = net.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level)
            pred_x0 = net.predict_start_from_noise(y_t, t=t, noise=pred_noise)

            snapshots_pred_noise[i] = pred_noise.cpu().squeeze(0).clone()
            snapshots_pred_x0[i] = pred_x0.cpu().squeeze(0).clone()

        # Actual p_sample step
        y_t = net.p_sample(y_t, t, y_cond=y_cond)

        if i in snapshot_steps:
            snapshots_yt[i] = y_t.cpu().squeeze(0).clone()

    # Final output
    snapshots_yt[-1] = y_t.cpu().squeeze(0).clone()

os.makedirs("debug_channels", exist_ok=True)

gt_cpu = gt.cpu().squeeze(0)

# --- Plot 1: y_t progression ---
all_steps = sorted(snapshots_yt.keys(), reverse=True)
n = len(all_steps)

fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
fig.suptitle("Reverse process: y_t from y_cond", fontsize=14)

for col, t_step in enumerate(all_steps):
    img = snapshots_yt[t_step]  # [1, H, W]
    img_vis = ((img[0].clamp(-1, 1) + 1) / 2).numpy()

    label = f"t={t_step}" if t_step >= 0 else "final"

    # Row 0: y_t
    axes[0, col].imshow(img_vis, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(label, fontsize=9)
    axes[0, col].axis("off")

    # Row 1: |y_t - GT| difference
    diff = np.abs(img_vis - ((gt_cpu[0].clamp(-1, 1) + 1) / 2).numpy())
    axes[1, col].imshow(diff, cmap="hot", vmin=0, vmax=0.3)
    axes[1, col].set_title(f"MAE={diff.mean():.4f}", fontsize=8)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("y_t", fontsize=11)
axes[1, 0].set_ylabel("|y_t - GT|", fontsize=11)

plt.tight_layout()
plt.savefig("debug_channels/reverse_from_cond_yt.png", dpi=150)
print("Saved debug_channels/reverse_from_cond_yt.png")

# --- Plot 2: Predicted noise at each snapshot ---
noise_steps = sorted(snapshots_pred_noise.keys(), reverse=True)
n2 = len(noise_steps)

fig2, axes2 = plt.subplots(2, n2, figsize=(2.5 * n2, 5))
fig2.suptitle("Predicted noise and predicted x0 at each step", fontsize=14)

for col, t_step in enumerate(noise_steps):
    noise = snapshots_pred_noise[t_step]  # [1, H, W]
    pred_x0 = snapshots_pred_x0[t_step]

    # Row 0: predicted noise
    noise_vis = noise[0].numpy()
    axes2[0, col].imshow(noise_vis, cmap="RdBu", vmin=-1, vmax=1)
    axes2[0, col].set_title(f"t={t_step}\nstd={noise_vis.std():.3f}", fontsize=8)
    axes2[0, col].axis("off")

    # Row 1: predicted x0
    x0_vis = ((pred_x0[0].clamp(-1, 1) + 1) / 2).numpy()
    axes2[1, col].imshow(x0_vis, cmap="gray", vmin=0, vmax=1)
    axes2[1, col].axis("off")

axes2[0, 0].set_ylabel("pred noise", fontsize=11)
axes2[1, 0].set_ylabel("pred x0", fontsize=11)

plt.tight_layout()
plt.savefig("debug_channels/reverse_from_cond_noise.png", dpi=150)
print("Saved debug_channels/reverse_from_cond_noise.png")

# --- Plot 3: Summary comparison ---
fig3, axes3 = plt.subplots(1, 5, figsize=(15, 3))
fig3.suptitle("Summary: Input → Output → GT", fontsize=14)

titles = ["Input (y_cond)", "t=75", "t=50", "Final output", "GT (clean)"]
images = [
    snapshots_yt[num_timesteps],
    snapshots_yt.get(75, snapshots_yt.get(74, snapshots_yt[num_timesteps])),
    snapshots_yt.get(50, snapshots_yt.get(49, snapshots_yt[num_timesteps])),
    snapshots_yt[-1],
    gt_cpu,
]

for col, (title, img) in enumerate(zip(titles, images)):
    img_vis = ((img[0].clamp(-1, 1) + 1) / 2).numpy()
    axes3[col].imshow(img_vis, cmap="gray", vmin=0, vmax=1)
    axes3[col].set_title(title, fontsize=10)
    axes3[col].axis("off")

plt.tight_layout()
plt.savefig("debug_channels/reverse_from_cond_summary.png", dpi=150)
print("Saved debug_channels/reverse_from_cond_summary.png")

# --- Print MSE at each snapshot ---
print("\n--- MSE from GT at each step (starting from y_cond) ---")
for t_step in all_steps:
    img = snapshots_yt[t_step]
    mse = ((img - gt_cpu) ** 2).mean().item()
    label = f"t={t_step}" if t_step >= 0 else "final"
    print(f"  {label:>10s}: MSE={mse:.6f}")

# --- Run SAME reverse process but starting from pure Gaussian noise ---
print("\n\nRunning reverse process from PURE GAUSSIAN NOISE for comparison...")
y_t_noise = torch.randn_like(y_cond).to(device)

snapshots_noise_start = {num_timesteps: y_t_noise.cpu().squeeze(0).clone()}

with torch.no_grad():
    for i in range(num_timesteps - 1, -1, -1):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        y_t_noise = net.p_sample(y_t_noise, t, y_cond=y_cond)
        if i in snapshot_steps:
            snapshots_noise_start[i] = y_t_noise.cpu().squeeze(0).clone()
    snapshots_noise_start[-1] = y_t_noise.cpu().squeeze(0).clone()

print("\n--- MSE from GT at each step (starting from Gaussian noise) ---")
noise_all_steps = sorted(snapshots_noise_start.keys(), reverse=True)
for t_step in noise_all_steps:
    img = snapshots_noise_start[t_step]
    mse = ((img - gt_cpu) ** 2).mean().item()
    label = f"t={t_step}" if t_step >= 0 else "final"
    print(f"  {label:>10s}: MSE={mse:.6f}")

# --- Plot 4: Side-by-side comparison ---
fig4, axes4 = plt.subplots(3, len(all_steps), figsize=(2.5 * len(all_steps), 7.5))
fig4.suptitle("y_t=y_cond (top) vs y_t=noise (mid) vs |difference| (bottom)", fontsize=14)

for col, t_step in enumerate(all_steps):
    label = f"t={t_step}" if t_step >= 0 else "final"

    # Row 0: from y_cond
    img_cond = ((snapshots_yt[t_step][0].clamp(-1, 1) + 1) / 2).numpy()
    axes4[0, col].imshow(img_cond, cmap="gray", vmin=0, vmax=1)
    axes4[0, col].set_title(label, fontsize=9)
    axes4[0, col].axis("off")

    # Row 1: from noise
    if t_step in snapshots_noise_start:
        img_noise = ((snapshots_noise_start[t_step][0].clamp(-1, 1) + 1) / 2).numpy()
    else:
        img_noise = np.zeros_like(img_cond)
    axes4[1, col].imshow(img_noise, cmap="gray", vmin=0, vmax=1)
    axes4[1, col].axis("off")

    # Row 2: absolute difference
    diff = np.abs(img_cond - img_noise)
    axes4[2, col].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
    axes4[2, col].set_title(f"diff={diff.mean():.4f}", fontsize=8)
    axes4[2, col].axis("off")

axes4[0, 0].set_ylabel("from y_cond", fontsize=11)
axes4[1, 0].set_ylabel("from noise", fontsize=11)
axes4[2, 0].set_ylabel("|diff|", fontsize=11)

plt.tight_layout()
plt.savefig("debug_channels/reverse_cond_vs_noise.png", dpi=150)
print("\nSaved debug_channels/reverse_cond_vs_noise.png")

# --- Final comparison printout ---
final_mse_cond = ((snapshots_yt[-1] - gt_cpu) ** 2).mean().item()
final_mse_noise = ((snapshots_noise_start[-1] - gt_cpu) ** 2).mean().item()
print(f"\n{'='*50}")
print(f"FINAL MSE (from y_cond):  {final_mse_cond:.6f}")
print(f"FINAL MSE (from noise):   {final_mse_noise:.6f}")
print(f"Difference:               {abs(final_mse_cond - final_mse_noise):.6f}")
print(f"Starting from y_cond {'helps' if final_mse_cond < final_mse_noise else 'hurts or is equal'}.")
print(f"{'='*50}")

plt.show()
print("\nDone!")
