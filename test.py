"""Test inference script for PromptIR_Simplified model."""

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import PromptIR_Simplified
from loaddata import PromptIRDataset


def reconstruct_from_patches(patches, patch_size=128, stride=64, full_size=256):
    """Reconstruct full image from overlapping patches."""
    device = patches.device
    channels = patches.shape[1]
    reconstructed = torch.zeros(
        (channels, full_size, full_size), device=device)
    weight = torch.zeros((channels, full_size, full_size), device=device)
    patch_idx = 0
    for y in range(0, full_size - patch_size + 1, stride):
        for x in range(0, full_size - patch_size + 1, stride):
            reconstructed[:, y:y + patch_size, x:x +
                          patch_size] += patches[patch_idx]
            weight[:, y:y + patch_size, x:x + patch_size] += 1.0
            patch_idx += 1
    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    reconstructed /= weight
    return reconstructed


def save_test_result(degraded, output, idx, save_dir):
    """Save side-by-side comparison image between degraded and output."""
    os.makedirs(save_dir, exist_ok=True)
    degraded = degraded.clamp(0, 1).cpu()
    output = output.clamp(0, 1).cpu()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(TF.to_pil_image(degraded))
    axs[0].set_title('Test Input')
    axs[1].imshow(TF.to_pil_image(output))
    axs[1].set_title('Test Output')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_sample_{idx}.png"))
    plt.close()


def save_test_comparison_image(output_img, save_path, idx):
    """Save a single output image."""
    output_img = output_img.clamp(0, 1).cpu()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(TF.to_pil_image(output_img))
    ax.set_title('Test Output')
    ax.axis('off')
    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"test_comparison_{idx}.png"))
    plt.close()
    print(f"Test image {idx} saved")


def run_test_inference(degraded_npz_path, model_path, save_dir='test_outputs', num_images=None):
    """Run inference on degraded input images and save results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = PromptIR_Simplified(
        inp_channels=3, out_channels=3,
        dim=48, num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded from checkpoint['model_state_dict']")
    else:
        model.load_state_dict(checkpoint)
        print("Model weights loaded directly from checkpoint")
    model.eval()

    patch_size = 128
    stride = 64
    full_size = 256

    degraded_data = np.load(degraded_npz_path)
    filenames = list(degraded_data.files)

    if num_images is None:
        num_images = len(filenames)
    else:
        num_images = min(num_images, len(filenames))

    print(f"Processing {num_images} images out of {len(filenames)} total images...")

    test_dataset = PromptIRDataset(
        degraded_npz_path=degraded_npz_path,
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=9,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    predictions_dict = {}
    print(f"Starting inference...")

    with torch.no_grad():
        image_idx = 0
        for batch_idx, degraded_patches in enumerate(tqdm(test_loader, desc="Processing images")):
            degraded_patches = degraded_patches.to(device)
            output_patches = model(degraded_patches)
            output_full = reconstruct_from_patches(
                output_patches, patch_size, stride, full_size)

            # 轉成 (H,W,3) uint8
            output_img_np = output_full.clamp(0, 1).cpu().numpy()
            output_img_np = (output_img_np * 255).astype(np.uint8)
            print(output_img_np.shape)

            filename = str(image_idx)+".png"
            predictions_dict[filename] = output_img_np
            print(f"Processed: {filename} -> shape: {output_img_np.shape}")

            vis_img = output_full.clamp(0, 1).cpu()
            save_test_comparison_image(vis_img, save_path=save_dir, idx=batch_idx)

            image_idx += 1


    pred_save_path = os.path.join(save_dir, 'pred.npz')
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(pred_save_path, **predictions_dict)

    print(f"\nPredictions saved to: {pred_save_path}")
    print(f"Contains {len(predictions_dict)} image predictions")
    print(f"Image format: (3, H, W) - Channels, Height, Width")
    print(f"Data type: uint8, range: 0-255")
    print(f"Visualization images saved to: {save_dir}")

    try:
        saved_data = np.load(pred_save_path)
        print(f"\nVerification: Successfully loaded {len(saved_data.files)} predicted images")
        for i, filename in enumerate(list(saved_data.files)[:3]):
            img_data = saved_data[filename]
            img_shape = img_data.shape
            img_dtype = img_data.dtype
            img_min = img_data.min()
            img_max = img_data.max()
            print(
                f"  - {filename}: shape={img_shape}, dtype={img_dtype}, range=[{img_min}, {img_max}]")
            if len(img_shape) == 3 and img_shape[0] == 3:
                print("    Correct format: (3, H, W)")
            else:
                print(f"    Wrong format! Expected (3, H, W), got {img_shape}")
        saved_data.close()
    except Exception as e:
        print(f"Error verifying saved file: {e}")

    return predictions_dict
