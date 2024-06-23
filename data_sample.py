import os
import cv2
import numpy as np
from edge_dataset import ParisStreetDataset

root = "/home/cvlab14/project/seongchan/edgegen/data/paris_street"
sample_root = "/home/cvlab14/project/seongchan/edgegen/data_sample"
os.makedirs(sample_root, exist_ok=True)

train_dataset = ParisStreetDataset(
    img_root=f"{root}/train",
    edge_root=f"{root}/train_edge",
    img_size=256,
)

n_samples = 5

for _ in range(n_samples):
    idx = np.random.randint(len(train_dataset))

    sample = train_dataset[idx]["images"]

    edge = sample[3:].permute(1, 2, 0).numpy() * 0.5 + 0.5
    edge = (edge * 255.0).astype(np.uint8)
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB)

    image = sample[:3].permute(1, 2, 0).numpy() * 0.5 + 0.5
    image = (image * 255.0).astype(np.uint8)

    y, x = np.where(edge[:, :, 0] > 0)
    overlay = image.copy()
    overlay[y, x] = np.array([255, 0, 0])

    image = np.concatenate([image, overlay, edge], axis=1)
    cv2.imwrite(f"{sample_root}/sample_{idx}.png", image[:, :, ::-1])