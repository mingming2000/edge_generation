
import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ParisStreetDataset(Dataset):
    def __init__(self, img_root=None, edge_root=None, img_size=256):
        super().__init__()
        self.img_root = img_root
        self.edge_root = edge_root
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize([0.5], [0.5])
        ])

        assert img_root is not None or edge_root is not None, "Both img_root and edge_root must be provided"

        if img_root is not None:
            self.imgs = os.listdir(img_root)
            self.imgs.sort()
        if edge_root is not None:
            self.edges = os.listdir(edge_root)
            self.edges.sort()

        if img_root is None:
            self.mode = 'edge'
        elif edge_root is None:
            self.mode = 'img'
        else:
            self.mode = 'all'
    
    def __len__(self):
        if self.mode == 'img' or self.mode == 'all':
            return len(self.imgs)
        else:
            return len(self.edges)
    
    def __getitem__(self, idx):
        if self.mode == 'img':
            img_path = os.path.join(self.img_root, self.imgs[idx])
            img = self.get_img(img_path)
        elif self.mode == 'edge':
            edge_path = os.path.join(self.edge_root, self.edges[idx])
            img = self.get_img(edge_path)
        else:
            img_path = os.path.join(self.img_root, self.imgs[idx])
            img = self.get_img(img_path)
            edge_path = os.path.join(self.edge_root, self.edges[idx])
            edge = self.get_img(edge_path)
            img = torch.cat([img, edge], dim=0)
        
        return {'images': self.transform(img)}

    def get_img(self, img_path):
        img = np.array(Image.open(img_path))
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).float() / 255.0
        if len(img.shape) == 2:
            img = img.unsqueeze(2)
        img = img.permute(2, 0, 1)
        assert img.shape[1:] == (self.img_size, self.img_size), f"Image shape is {img.shape}"
        return img