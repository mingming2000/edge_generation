import os
import imageio.v2 as iio
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
import numpy as np
import torch
from tqdm.auto import tqdm

# CUDA_VISIBLE_DEVICES=4 python controlnet_inpaint.py
img_dir = "/home/cvlab14/project/seongchan/edgegen/data/paris_street/eval"

############################################################################################################

# TODO just change this

mask_version = "S"
model_type = 'ddpm_1c'

############################################################################################################

model_version = f"{model_type}_{mask_version}"
edge_dir = f"/home/cvlab14/project/seongchan/edgegen/edge_results/{model_version}"
save_dir = f"/home/cvlab14/project/seongchan/edgegen/outpaint_results/{model_version}"
mask_dir = f"/home/cvlab14/project/seongchan/edgegen/data/paris_street/eval_mask_v2/{mask_version}"
edge_list = os.listdir(edge_dir)
edge_list.sort()

os.makedirs(save_dir, exist_ok=True)
img_list = os.listdir(img_dir)
mask_list = os.listdir(mask_dir)
img_list.sort()
mask_list.sort()

prompt = "paris street view"

if prompt == "":
    os.makedirs(os.path.join(save_dir, "no_prompt"), exist_ok=True)
    save_dir = os.path.join(save_dir, "no_prompt")
else:
    os.makedirs(os.path.join(save_dir, prompt), exist_ok=True)
    save_dir = os.path.join(save_dir, prompt)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
generator = torch.Generator(device="cpu").manual_seed(1)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

for idx, (img, mask, edge) in enumerate(zip(tqdm(img_list), mask_list, edge_list)):
    img = Image.open(os.path.join(img_dir, img))
    img = img.resize((512, 512))
    mask = iio.imread(os.path.join(mask_dir, mask)) / 255.0
    mask = Image.fromarray(((1 - mask) * 255).astype(np.uint8))
    mask = mask.resize((512, 512))
    edge = Image.open(os.path.join(edge_dir, edge))
    edge = edge.resize((512, 512))

    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        generator=generator,
        eta=1.0,
        image=img,
        mask_image=mask,
        control_image=edge,
    ).images[0]

    image.save(os.path.join(save_dir, f"{idx:04d}.png"))