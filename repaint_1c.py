
import os
import cv2
import torch
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import imageio.v2 as iio
from diffusers import DDPMScheduler, UNet2DModel
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = "/home/cvlab14/project/seongchan/edgegen/output/1channels_1"
eval_out_dir = "/home/cvlab14/project/seongchan/edgegen/visualize/1channels_t"
data_dir = "/home/cvlab14/project/seongchan/edgegen/data/paris_street"

os.makedirs(eval_out_dir, exist_ok=True)

# model and scheduler
unet = UNet2DModel.from_pretrained(f"{save_dir}/unet").to(device)
scheduler = DDPMScheduler.from_pretrained(f"{save_dir}/scheduler")

transform = transforms.Normalize(mean=[0.5], std=[0.5])

edge_list = os.listdir(os.path.join(data_dir, "eval_edge"))
mask_list = os.listdir(os.path.join(data_dir, "eval_mask"))

edge_list.sort()
mask_list.sort()

for idx, (edge, mask) in enumerate(zip(edge_list, mask_list)):
    edge = iio.imread(os.path.join(data_dir, "eval_edge", edge))
    mask = iio.imread(os.path.join(data_dir, "eval_mask", mask))

    edge = transform(torch.tensor(edge).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

    clean_edge = edge * mask

    input = torch.randn_like(clean_edge).to(device)
    for timesteps, t in enumerate(tqdm(scheduler.timesteps)):
        with torch.no_grad():
            noise_pred = unet(input, t).sample
        input = scheduler.step(noise_pred, t, input).prev_sample

        if t > 0:
            noisy_edge = scheduler.add_noise(
                original_samples=clean_edge,
                noise=torch.randn_like(clean_edge).to(device),
                timesteps=t,
            )
            input = input * (1 - mask) + noisy_edge * mask
    inpainted_edge = ((input / 2 + 0.5).clamp(0, 1).cpu().squeeze().numpy() > 0.5)
    inpainted_edge = Image.fromarray((inpainted_edge * 255).astype(np.uint8))
    inpainted_edge.save(f"{eval_out_dir}/inpaint_{idx}.png")