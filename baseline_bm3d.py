"""
Compare BM3D denoising against clean GT using the same metrics as the diffusion model.
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import bm3d
from models.metric import mse, mae, psnr, ssim, gmsd

clean_dir = "data/test_clean"
noisy_dir = "data/test_noisy"

tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

tfs_no_norm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Try a few sigma values since we don't know the exact noise level
sigma_values = [0.01, 0.02, 0.05, 0.1, 0.2]

files = sorted(os.listdir(clean_dir))

for sigma_est in sigma_values:
    metrics = {'mse': [], 'mae': [], 'psnr': [], 'ssim': [], 'gmsd': []}

    for fname in files:
        # Load as [0, 1] for BM3D
        noisy_01 = tfs_no_norm(Image.open(os.path.join(noisy_dir, fname)).convert('L')).squeeze().numpy()

        # BM3D expects [0, 1] input and sigma in [0, 1] scale
        denoised_01 = bm3d.bm3d(noisy_01, sigma_psd=sigma_est, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        denoised_01 = np.clip(denoised_01, 0, 1)

        # Convert to [-1, 1] for metrics (same as model pipeline)
        clean = tfs(Image.open(os.path.join(clean_dir, fname)).convert('L')).unsqueeze(0)
        denoised = torch.from_numpy(denoised_01).float().unsqueeze(0).unsqueeze(0) * 2 - 1

        metrics['mse'].append(mse(denoised, clean).item())
        metrics['mae'].append(mae(denoised, clean).item())
        metrics['psnr'].append(psnr(denoised, clean).item())
        metrics['ssim'].append(ssim(denoised, clean).item())
        metrics['gmsd'].append(gmsd(denoised, clean).item())

    print(f"\nBM3D (sigma_est={sigma_est})")
    print('='*50)
    for key, values in metrics.items():
        print('{}: {:.6f}'.format(key, np.mean(values)))

print('\n' + '='*50)
print('For reference -- no model baseline:')
print('mse: 0.011899, mae: 0.052739, psnr: 25.495590, ssim: 0.857914, gmsd: 0.129907')
print('='*50)

# --- Visual comparison using best sigma ---
import matplotlib.pyplot as plt

best_sigma = 0.05
os.makedirs("bm3d_results", exist_ok=True)

n_show = min(8, len(files))
fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
fig.suptitle(f"Noisy (top) vs BM3D sigma={best_sigma} (mid) vs Clean GT (bottom)", fontsize=14)

for col, fname in enumerate(files[:n_show]):
    noisy_01 = tfs_no_norm(Image.open(os.path.join(noisy_dir, fname)).convert('L')).squeeze().numpy()
    clean_01 = tfs_no_norm(Image.open(os.path.join(clean_dir, fname)).convert('L')).squeeze().numpy()
    denoised_01 = bm3d.bm3d(noisy_01, sigma_psd=best_sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    denoised_01 = np.clip(denoised_01, 0, 1)

    axes[0, col].imshow(noisy_01, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(fname[:12], fontsize=7)
    axes[0, col].axis("off")

    axes[1, col].imshow(denoised_01, cmap="gray", vmin=0, vmax=1)
    axes[1, col].axis("off")

    axes[2, col].imshow(clean_01, cmap="gray", vmin=0, vmax=1)
    axes[2, col].axis("off")

    # Save individual BM3D results
    out_pil = Image.fromarray((denoised_01 * 255).astype(np.uint8), mode="L")
    out_pil.save(os.path.join("bm3d_results", fname))

axes[0, 0].set_ylabel("Noisy", fontsize=11)
axes[1, 0].set_ylabel("BM3D", fontsize=11)
axes[2, 0].set_ylabel("Clean GT", fontsize=11)

plt.tight_layout()
plt.savefig("bm3d_results/comparison_grid.png", dpi=150)
print(f"\nSaved bm3d_results/comparison_grid.png and {len(files)} individual images")
plt.show()
