
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

save_dir = "/home/cvlab14/project/seongchan/edgegen/output/4channels_1"
eval_out_dir = "/home/cvlab14/project/seongchan/edgegen/visualize/4channels_t-1"
data_dir = "/home/cvlab14/project/seongchan/edgegen/data/paris_street"

os.makedirs(eval_out_dir, exist_ok=True)

# model and scheduler
unet = UNet2DModel.from_pretrained(f"{save_dir}/unet").to(device)
scheduler = DDPMScheduler.from_pretrained(f"{save_dir}/scheduler")

transform = transforms.Normalize(mean=[0.5], std=[0.5])

img_list = os.listdir(os.path.join(data_dir, "eval"))
edge_list = os.listdir(os.path.join(data_dir, "eval_edge"))
mask_list = os.listdir(os.path.join(data_dir, "eval_mask"))

img_list.sort()
edge_list.sort()
mask_list.sort()

for idx, (img, edge, mask) in enumerate(zip(img_list, edge_list, mask_list)):
    img = iio.imread(os.path.join(data_dir, "eval", img))
    img = cv2.resize(img, (256, 256))
    edge = iio.imread(os.path.join(data_dir, "eval_edge", edge))
    mask = iio.imread(os.path.join(data_dir, "eval_mask", mask))

    img = transform(torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0)
    edge = transform(torch.tensor(edge).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0)
    mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

    clean_img = torch.cat([img, edge], dim=1) * mask

    input = torch.randn_like(clean_img).to(device)
    for timesteps, t in enumerate(tqdm(scheduler.timesteps)):
        with torch.no_grad():
            noise_pred = unet(input, t).sample
        input = scheduler.step(noise_pred, t, input).prev_sample

        if t > 0:
            noisy_img = scheduler.add_noise(
                original_samples=clean_img,
                noise=torch.randn_like(clean_img).to(device),
                timesteps=t - 1,
            )
            input = input * (1 - mask) + noisy_img * mask
    inpainted_img = (input[0, :3, :, :] / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    inpainted_edge = ((input[0, 3:, :, :] / 2 + 0.5).clamp(0, 1).cpu().squeeze().numpy() > 0.5)
    inpainted_img = Image.fromarray((inpainted_img * 255).astype(np.uint8))
    inpainted_edge = Image.fromarray((inpainted_edge * 255).astype(np.uint8))
    inpainted_img.save(f"{eval_out_dir}/inpained_img_{idx}.png")
    inpainted_edge.save(f"{eval_out_dir}/inpained_edge_{idx}.png")