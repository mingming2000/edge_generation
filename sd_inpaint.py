import os
import cv2
import torch
import numpy as np
from tqdm.auto import tqdm
import imageio.v2 as iio
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
).to("cuda")

img_dir = "/home/cvlab14/project/seongchan/edgegen/data/paris_street/eval"
mask_root = "/home/cvlab14/project/seongchan/edgegen/data/paris_street/eval_mask_v2"
save_root = "/home/cvlab14/project/seongchan/edgegen/sd_inpaint"

mask_types = ["L", "M", "S"]

for mask_type in mask_types:
    mask_dir = os.path.join(mask_root, mask_type)

    prompt = "paris street view"

    save_dir = os.path.join(save_root, prompt, mask_type)
    os.makedirs(save_dir, exist_ok=True)

    img_list = os.listdir(img_dir)
    img_list.sort()
    mask_list = os.listdir(mask_dir)
    mask_list.sort()

    for idx, (img, mask) in enumerate(zip(tqdm(img_list), mask_list)):
        img = Image.open(os.path.join(img_dir, img))
        mask = (np.array(Image.open(os.path.join(mask_dir, mask))) > 0).astype(np.uint8)
        mask = Image.fromarray((1 - mask) * 255)
        inpainted_img = pipe(prompt=prompt, image=img, mask_image=mask).images[0]
        inpainted_img.save(os.path.join(save_dir, f"{idx:04d}.png"))