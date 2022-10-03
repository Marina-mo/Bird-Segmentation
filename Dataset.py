#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os



class MyDataset(Dataset):
    def __init__(self, input_path, gt_path, img_preprocess=None, gt_preprocess=None):
        classes = os.listdir(input_path)
        self._input_images = []
        self._target_masks = []
        for cls in classes:
            imgs = os.listdir(os.path.join(input_path, cls))
            for img in imgs:
                self._input_images.append(os.path.join(input_path, cls, img))
                self._target_masks.append(os.path.join(gt_path, cls, img[:-3]+'png'))

        self._img_preprocess = img_preprocess
        self._gt_preprocess = gt_preprocess
        
    def __len__(self):
        return len(self._input_images)

    def __getitem__(self, idx):
        
        image = cv2.imread(self._input_images[idx])
        mask = cv2.imread(self._target_masks[idx], cv2.IMREAD_GRAYSCALE)

        image = self._img_preprocess(image)
        mask = self._gt_preprocess(mask)

        return [image, mask]


