"""
Run KDLAE-Teacher on our test set and compare metrics against GT.
Also runs No Model and BM3D for a complete comparison table.
"""
import sys
sys.path.insert(0, r"C:\Users\ramah\OneDrive\Desktop\Dfki_ws\SIMNFND\Rethink_Acoustic_Image_Enhancement\KDLAE")

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import bm3d
import cv2
from models.metric import mse, mae, psnr, ssim, gmsd
from KDLAE_model import KDLAE_teacher

clean_dir = "data/test_clean"
noisy_dir = "data/test_noisy"
model_dir = "experiments/test_denoising_experiment_260321_020518/results/test/0"

# --- Load KDLAE-Teacher ---
weights_path = r"C:\Users\ramah\OneDrive\Desktop\Dfki_ws\SIMNFND\Rethink_Acoustic_Image_Enhancement\KDLAE\weights\KDLAE_T.pth"

kdlae_model = KDLAE_teacher(
    inp_channels=3,
    out_channels=3,
    dim=48,
    num_blocks=[4, 6, 6, 8],
    num_refinement_blocks=4,
    heads=[1, 2, 4, 8],
    ffn_expansion_factor=2.66,
    bias=False,
    LayerNorm_type="BiasFree",
    dual_pixel_task=False,
    static="train",
    params="cat",
)

checkpoint = torch.load(weights_path, map_location="cpu")
kdlae_model.load_state_dict(checkpoint['params'])
kdlae_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kdlae_model = kdlae_model.to(device)
print(f"KDLAE-Teacher loaded on {device}")

# --- Transforms ---
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
    'KDLAE-T (dr=0.5)': {m: [] for m in metric_names},
    'Palette (ours)': {m: [] for m in metric_names},
}

files = sorted(os.listdir(clean_dir))
img_multiple_of = 8
denoise_rate = 0.5

print(f"Processing {len(files)} test images...")

for idx, fname in enumerate(files):
    clean_path = os.path.join(clean_dir, fname)
    noisy_path = os.path.join(noisy_dir, fname)
    model_path = os.path.join(model_dir, f"Out_{fname}")

    # Load for metrics ([-1, 1] range)
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

    # --- KDLAE-Teacher ---
    with torch.no_grad():
        # Load as RGB (replicate grayscale to 3 channels)
        img_gray = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        img_gray = cv2.resize(img_gray, (256, 256))
        img_rgb = np.stack([img_gray, img_gray, img_gray], axis=2)
        img_float = img_rgb.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_padded = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')

        alpha = torch.ones((1, 1, input_padded.shape[2], input_padded.shape[3]), device=device) * denoise_rate

        pred = kdlae_model({'img': input_padded, 'denoise_rate': alpha})
        restored = torch.clamp(pred['hq'], 0, 1)
        restored = restored[:, :, :h, :w]

        # Take first channel (grayscale), convert to [-1, 1]
        kdlae_gray = restored[:, 0:1, :, :].cpu() * 2 - 1

    for name, fn in zip(metric_names, metric_fns):
        results['KDLAE-T (dr=0.5)'][name].append(fn(kdlae_gray, clean).item())

    # --- Palette model ---
    if os.path.exists(model_path):
        model_out = tfs(Image.open(model_path).convert('L')).unsqueeze(0)
        for name, fn in zip(metric_names, metric_fns):
            results['Palette (ours)'][name].append(fn(model_out, clean).item())

    if (idx + 1) % 10 == 0:
        print(f"  {idx + 1}/{len(files)} done")

# --- Print comparison table ---
print('\n' + '='*78)
print(f'{"Method":<25} {"MSE":>8} {"MAE":>8} {"PSNR":>8} {"SSIM":>8} {"GMSD":>8}')
print('='*78)
for method, metrics in results.items():
    vals = [np.mean(metrics[m]) for m in metric_names]
    print(f'{method:<25} {vals[0]:>8.6f} {vals[1]:>8.6f} {vals[2]:>8.4f} {vals[3]:>8.6f} {vals[4]:>8.6f}')
print('='*78)
