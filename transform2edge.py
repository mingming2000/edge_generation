
import os
import cv2
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
data_dir = "/home/seongchan/project/edgegen/data/paris_street"

split = ['train', 'eval']

for s in split:
    print(f"Processing {s} data...")
    os.makedirs(os.path.join(data_dir, s + "_edge"), exist_ok=True)
    data_list = os.listdir(os.path.join(data_dir, s))
    for data_path in tqdm(data_list):
        if data_path.endswith("_edge.png"):
            os.remove(os.path.join(data_dir, s, data_path))
            continue
        data = os.path.join(data_dir, s, data_path)
        data = cv2.imread(data)
        data = cv2.resize(data, (256, 256))
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
        data = cv2.Canny(data, 100, 200)
        data = Image.fromarray(data)
        data.save(os.path.join(data_dir, s + "_edge", data_path[:-4] + "_edge.png"))