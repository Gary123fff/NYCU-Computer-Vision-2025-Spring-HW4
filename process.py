"""Module for processing and patching clean/degraded images for training/testing."""

import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def process_test_degraded_images(base_path='hw4_realse_dataset/test',
                                 resize_shape=(256, 256),
                                 patch_size=128,
                                 stride=64,
                                 sample_num=0):
    """Process degraded test images by resizing and cutting overlapping patches."""
    degraded_path = os.path.join(base_path, 'degraded')
    degraded_files = sorted([
        f for f in os.listdir(degraded_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    degraded_patches = {}

    print("Processing degraded images, resizing and cutting 9 overlapping patches each...")

    for d_file in tqdm(degraded_files, desc="Processing Test Degraded"):
        degraded_img = Image.open(os.path.join(
            degraded_path, d_file)).convert('RGB')
        degraded_img = degraded_img.resize(
            resize_shape, Image.Resampling.BICUBIC)
        degraded_arr = np.array(degraded_img)

        patch_id = 0
        for y in range(0, resize_shape[0] - patch_size + 1, stride):
            for x in range(0, resize_shape[1] - patch_size + 1, stride):
                degraded_patch = degraded_arr[y:y +
                                              patch_size, x:x + patch_size, :]
                degraded_patch = np.transpose(degraded_patch, (2, 0, 1))
                patch_key = f"{d_file}_patch{patch_id}"
                degraded_patches[patch_key] = degraded_patch
                patch_id += 1

    print(f"Total {len(degraded_patches)} patches created for test set.")

    save_dir = 'output_npz_patches_test'
    os.makedirs(save_dir, exist_ok=True)

    np.savez(os.path.join(save_dir, 'test_degraded_patches.npz'),
             **degraded_patches)
    print(f"Saved test degraded patch .npz file to: {save_dir}")

    if sample_num > 0:
        print(f"\nShowing {sample_num} random degraded patches:")
        patch_keys = list(degraded_patches.keys())
        samples = random.sample(patch_keys, sample_num)

        for key in samples:
            degraded_patch = np.transpose(degraded_patches[key], (1, 2, 0))
            plt.figure(figsize=(3, 3))
            plt.imshow(degraded_patch)
            plt.title(f"Degraded\n{key}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()


def process_and_show_images(base_path='hw4_realse_dataset/train',
                            resize_shape=(256, 256),
                            patch_size=128,
                            stride=64,
                            sample_num=5):
    """Process paired clean and degraded images and show sampled patch pairs."""
    clean_path = os.path.join(base_path, 'clean')
    degraded_path = os.path.join(base_path, 'degraded')

    clean_files = sorted([
        f for f in os.listdir(clean_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    degraded_files = sorted([
        f for f in os.listdir(degraded_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    assert len(clean_files) == len(degraded_files), (
        f"Mismatch in image count: clean={len(clean_files)}, degraded={len(degraded_files)}"
    )

    clean_patches = {}
    degraded_patches = {}

    print("Processing images, resizing and cutting 9 overlapping patches each...")

    for c_file, d_file in tqdm(zip(clean_files, degraded_files),
                               total=len(clean_files),
                               desc="Processing"):
        clean_img = Image.open(os.path.join(clean_path, c_file)).convert('RGB')
        clean_img = clean_img.resize(resize_shape, Image.Resampling.BICUBIC)

        degraded_img = Image.open(os.path.join(
            degraded_path, d_file)).convert('RGB')
        degraded_img = degraded_img.resize(
            resize_shape, Image.Resampling.BICUBIC)

        clean_arr = np.array(clean_img)
        degraded_arr = np.array(degraded_img)

        patch_id = 0
        for y in range(0, resize_shape[0] - patch_size + 1, stride):
            for x in range(0, resize_shape[1] - patch_size + 1, stride):
                clean_patch = clean_arr[y:y + patch_size, x:x + patch_size, :]
                degraded_patch = degraded_arr[y:y +
                                              patch_size, x:x + patch_size, :]

                clean_patch = np.transpose(clean_patch, (2, 0, 1))
                degraded_patch = np.transpose(degraded_patch, (2, 0, 1))

                patch_key = f"{d_file}_patch{patch_id}"
                clean_patches[patch_key] = clean_patch
                degraded_patches[patch_key] = degraded_patch
                patch_id += 1

    print(f"Total {len(clean_patches)} patches created.")

    save_dir = 'output_npz_patches'
    os.makedirs(save_dir, exist_ok=True)

    np.savez(os.path.join(save_dir, 'train_clean_patches.npz'), **clean_patches)
    np.savez(os.path.join(save_dir, 'train_degraded_patches.npz'),
             **degraded_patches)
    print(f"Saved patch .npz files in folder: {save_dir}")

    if sample_num > 0:
        print(
            f"\nShowing {sample_num} random patch pairs (Degraded vs Clean):")
        patch_keys = list(clean_patches.keys())
        samples = random.sample(patch_keys, sample_num)

        for key in samples:
            clean_patch = np.transpose(clean_patches[key], (1, 2, 0))
            degraded_patch = np.transpose(degraded_patches[key], (1, 2, 0))

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(degraded_patch)
            plt.title(f"Degraded\n{key}")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(clean_patch)
            plt.title(f"Clean\n{key}")
            plt.axis('off')

            plt.tight_layout()
            plt.show()
