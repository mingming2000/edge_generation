# Code Description

`transform2edge.py` - cv2.Canny를 이용하여 이미지로부터 edge 이미지를 생성하는 코드

`edge_dataset.py` - `train.py`에서 사용하는 dataset class 코드

`data_sample.py` - dataset 내 샘플 시각화 코드

`train.py` - edge generate diffusion model을 학습하는 코드, option으로 channel이 있고, channel은 1과 4를 지원한다.

`sd_inpaint.py` - stable diffusion inpainting 모델을 통해 eval set에 대해 inpainting을 수행하는 코드

`repaint_4c.py` - 4 channel edge generate diffusion model을 통해 eval set에 대해 inpainting을 수행하는 코드

`repaint_1c.py` - 1 channel edge generate diffusion model을 통해 eval set에 대해 inpainting을 수행하는 코드

`metric.py` - FID, IS, LPIPS metric을 계산하는 코드

`controlnet_inpaint.py` - `repaint_4c.py` 또는 `repaint_1c.py`를 통해 생성된 edge를 controlnet을 통해 inpainting하는 코드