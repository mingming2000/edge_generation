import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

gt_dir = "/home/cvlab14/project/seongchan/edgegen/data/paris_street/eval"

gen_root = "/home/cvlab14/project/seongchan/edgegen/outpaint_results"
model_types = ["basegan", "ddpm_4c", "ddpm_1c"]
mask_types = ["L", "S"]

gen_dirs = []
for model_type in model_types:
    for mask_type in mask_types:
        gen_dirs.append(os.path.join(gen_root, f"{model_type}_{mask_type}"))

# gen_dirs = ["/home/cvlab14/project/seongchan/edgegen/controlnet_inpaint_results_v1/attngan_final/paris street view"]
# for mask_type in mask_types:
#     gen_dirs.append(os.path.join("/home/cvlab14/project/seongchan/edgegen/sd_inpaint/paris street view", f"sd_{mask_type}"))

prompt = "paris street view"

for gen_dir in gen_dirs:
    print(gen_dir.split("/")[-1])
    gen_dir = os.path.join(gen_dir, prompt)

    gt_list = os.listdir(gt_dir)
    gt_list.sort()

    gen_list = os.listdir(gen_dir)
    gen_list.sort()

    fid = FrechetInceptionDistance(feature=64)
    inception = InceptionScore()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    gts = [torch.tensor(cv2.resize(np.array(Image.open(os.path.join(gt_dir, gt))), (256, 256), interpolation=cv2.INTER_LINEAR), dtype=torch.uint8).permute(2, 0, 1) for gt in gt_list]
    gts = torch.stack(gts)
    gens = [torch.tensor(cv2.resize(np.array(Image.open(os.path.join(gen_dir, gen))), (256, 256), interpolation=cv2.INTER_LINEAR), dtype=torch.uint8).permute(2, 0, 1) for gen in gen_list]
    gens = torch.stack(gens)

    fid.update(gts, real=True)
    fid.update(gens, real=False)
    print(f'  FID : {fid.compute()}')

    inception.update(gens)
    print(f'   IS : {inception.compute()}')

    gts = (gts / 255.0) * 2 - 1
    gens = (gens / 255.0) * 2 - 1
    print(f'LPIPS : {lpips(gts, gens)}')

'''
FID는 낮을수록 좋다
IS는 높을수록 좋다
LPIPS는 낮을수록 좋다

GT
- FID: 7.048583938740194e-12
- IS: (tensor(3.3272), tensor(0.4148))
- LPIPS: 0.0

Diffusion-based [Stable Diffusion Inpainting]
- FID: 1.6236331462860107
- IS: (tensor(2.8232), tensor(0.3943))
- LPIPS: 0.2727859914302826

Diffusion-based [DDPM]
- 4channels_t-1
    - FID: 0.8033306002616882
    - IS: (tensor(2.5827), tensor(0.4492))
    - LPIPS: 0.3352722227573395
- 1channels_t-1
    - FID: 0.8716163635253906
    - IS: (tensor(2.6646), tensor(0.6041))
    - LPIPS: 0.3330436944961548

GAN-based
- Attngan
    - FID: 0.7469877004623413 *** Best FID
    - IS: (tensor(2.8030), tensor(0.5865)) *** Best IS
    - LPIPS: 0.3311452567577362a
- Baseline
    - FID: 0.7960066199302673
    - IS: (tensor(2.7355), tensor(0.5522))
    - LPIPS: 0.3271228075027466 *** Best LPIPS
- Outpaint_lr
    - FID: 0.780973494052887
    - IS: (tensor(2.5873), tensor(0.5289))
    - LPIPS: 0.3289518654346466
- Outpaint_mask_v2_weight
    - FID: 0.8013520240783691
    - IS: (tensor(2.6418), tensor(0.3595))
    - LPIPS: 0.3353106677532196
- Outpaint_weight
    - FID: 0.8766368627548218
    - IS: (tensor(2.5455), tensor(0.5168))
    - LPIPS: 0.32842811942100525
'''