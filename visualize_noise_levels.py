import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import os

# Load config
with open("config/denoising_my_data.json") as f:
    config = json.load(f)

beta_config = config["model"]["which_networks"][0]["args"]["beta_schedule"]["train"]
n_timestep = beta_config["n_timestep"]
linear_start = beta_config["linear_start"]
linear_end = beta_config["linear_end"]

# Build noise schedule
betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
alphas = 1. - betas
gammas = np.cumprod(alphas, axis=0)

# Load a sample clean image
clean_dir = "data/train_clean"
sample_file = sorted(os.listdir(clean_dir))[0]
img = Image.open(os.path.join(clean_dir, sample_file)).convert('L').resize((256, 256))
tfs = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
y_0 = tfs(img).unsqueeze(0)

# Pick evenly spaced timesteps
num_samples = min(10, n_timestep)
timesteps = np.linspace(0, n_timestep - 1, num_samples, dtype=int)

noise = torch.randn_like(y_0)

images = []
for t in timesteps:
    gamma_t = torch.tensor(gammas[t], dtype=torch.float32)
    y_t = gamma_t.sqrt() * y_0 + (1 - gamma_t).sqrt() * noise
    # Convert back to [0, 1] for saving
    y_t_img = (y_t.squeeze().clamp(-1, 1) + 1) / 2
    y_t_img = (y_t_img.numpy() * 255).astype(np.uint8)
    images.append(y_t_img)
    print(f"t={t:4d} | gamma={gammas[t]:.6f} | signal={gammas[t]:.1%} | noise={1-gammas[t]:.1%}")

# Save as a horizontal strip
strip = np.concatenate(images, axis=1)
Image.fromarray(strip).save("noise_levels.jpg")
print(f"\nSaved noise_levels.jpg ({num_samples} images at timesteps {list(timesteps)})")
