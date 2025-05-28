"""
This module defines the PromptIRDataset class for loading and augmenting
infrared image datasets from .npz files for deep learning applications.
"""

import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class PromptIRDataset(Dataset):
    """
    A PyTorch Dataset for infrared image restoration tasks.

    Parameters:
        degraded_npz_path (str): Path to .npz file containing degraded images.
        clean_npz_path (str, optional): Path to .npz file containing clean ground-truth images.
        augment (bool): Whether to apply data augmentation.
    """

    def __init__(self, degraded_npz_path, clean_npz_path=None, augment=False):
        with np.load(degraded_npz_path) as degraded_data:
            self.degraded_data = {k: degraded_data[k] for k in degraded_data}

        if clean_npz_path is not None:
            with np.load(clean_npz_path) as clean_data:
                self.clean_data = {k: clean_data[k] for k in clean_data}
            self.with_gt = True
        else:
            self.clean_data = None
            self.with_gt = False

        self.keys = list(self.degraded_data.keys())
        self.augment = augment

        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(p=0.2),
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2(transpose_mask=True)
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
                ToTensorV2(transpose_mask=True)
            ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        degraded_img = self.degraded_data[key] / 255.0

        if degraded_img.shape[0] == 3:
            degraded_img = np.transpose(degraded_img, (1, 2, 0))

        if self.with_gt:
            clean_img = self.clean_data[key] / 255.0
            if clean_img.shape[0] == 3:
                clean_img = np.transpose(clean_img, (1, 2, 0))

            augmented = self.transform(image=degraded_img, mask=clean_img)
            return augmented["image"], augmented["mask"]

        augmented = self.transform(image=degraded_img)
        return augmented["image"]
