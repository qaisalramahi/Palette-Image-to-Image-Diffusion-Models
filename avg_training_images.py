import os
import numpy as np
from PIL import Image

clean_dir = "data/train_clean"
output_path = "mean_training_image.jpg"

images = []
for fname in sorted(os.listdir(clean_dir)):
    img = np.array(Image.open(os.path.join(clean_dir, fname)).convert('L').resize((256, 256)), dtype=np.float64)
    images.append(img)

mean_img = np.mean(images, axis=0).astype(np.uint8)
Image.fromarray(mean_img).save(output_path)
print(f"Saved mean of {len(images)} images to {output_path}")
