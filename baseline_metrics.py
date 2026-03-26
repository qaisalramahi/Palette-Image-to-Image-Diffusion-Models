"""
Compare metrics across: No Model, BM3D, and Palette (diffusion model).
All evaluated on the same test set against clean GT.
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
model_dir = "experiments/test_denoising_experiment_260321_020518/results/test/0"

tfs = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

tfs_no_norm = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

metric_names = ['mse', 'mae', 'psnr', 'ssim', 'gmsd']
metric_fns = [mse, mae, psnr, ssim, gmsd]

results = {
    'No Model': {m: [] for m in metric_names},
    'BM3D (sigma=0.05)': {m: [] for m in metric_names},
    'Palette (ours)': {m: [] for m in metric_names},
}

files = sorted(os.listdir(clean_dir))

for fname in files:
    clean_path = os.path.join(clean_dir, fname)
    noisy_path = os.path.join(noisy_dir, fname)
    model_path = os.path.join(model_dir, f"Out_{fname}")

    clean = tfs(Image.open(clean_path).convert('L')).unsqueeze(0)
    noisy = tfs(Image.open(noisy_path).convert('L')).unsqueeze(0)

    # --- No Model ---
    for name, fn in zip(metric_names, metric_fns):
        results['No Model'][name].append(fn(noisy, clean).item())

    # --- BM3D ---
    noisy_01 = tfs_no_norm(Image.open(noisy_path).convert('L')).squeeze().numpy()
    denoised_01 = bm3d.bm3d(noisy_01, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    denoised_01 = np.clip(denoised_01, 0, 1)
    bm3d_tensor = torch.from_numpy(denoised_01).float().unsqueeze(0).unsqueeze(0) * 2 - 1
    for name, fn in zip(metric_names, metric_fns):
        results['BM3D (sigma=0.05)'][name].append(fn(bm3d_tensor, clean).item())

    # --- Palette model ---
    if os.path.exists(model_path):
        model_out = tfs(Image.open(model_path).convert('L')).unsqueeze(0)
        for name, fn in zip(metric_names, metric_fns):
            results['Palette (ours)'][name].append(fn(model_out, clean).item())
    else:
        print(f"Warning: missing model output for {fname}")

# --- Print comparison table ---
print('\n' + '='*70)
print(f'{"Method":<25} {"MSE":>8} {"MAE":>8} {"PSNR":>8} {"SSIM":>8} {"GMSD":>8}')
print('='*70)
for method, metrics in results.items():
    vals = [np.mean(metrics[m]) for m in metric_names]
    print(f'{method:<25} {vals[0]:>8.6f} {vals[1]:>8.6f} {vals[2]:>8.4f} {vals[3]:>8.6f} {vals[4]:>8.6f}')
print('='*70)
