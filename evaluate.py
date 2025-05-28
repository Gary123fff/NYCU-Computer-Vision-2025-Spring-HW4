"""evaluate.py - Evaluate a trained image restoration model and visualize results."""

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from loaddata import PromptIRDataset
from model import PromptIR_Simplified


def reconstruct_from_patches(patches, patch_size=128, stride=64, full_size=256):
    """
    Reconstruct a full image from overlapping patches.

    Args:
        patches (Tensor): Tensor of shape [N, C, H, W].
        patch_size (int): Size of each patch.
        stride (int): Stride used when creating patches.
        full_size (int): Final image size.

    Returns:
        Tensor: The reconstructed full-size image.
    """
    device = patches.device
    channels = patches.shape[1]
    reconstructed = torch.zeros(
        (channels, full_size, full_size), device=device)
    weight = torch.zeros((channels, full_size, full_size), device=device)

    patch_idx = 0
    for y in range(0, full_size - patch_size + 1, stride):
        for x in range(0, full_size - patch_size + 1, stride):
            reconstructed[:, y:y+patch_size, x:x +
                          patch_size] += patches[patch_idx]
            weight[:, y:y+patch_size, x:x+patch_size] += 1.0
            patch_idx += 1

    weight = torch.where(weight == 0, torch.ones_like(weight), weight)
    reconstructed /= weight
    return reconstructed


def save_comparison_image(output_img, clean_img, degraded_img, save_path, idx):
    """
    Save a side-by-side comparison of degraded, output, and ground truth images.

    Args:
        output_img (Tensor): Restored image.
        clean_img (Tensor): Ground truth image.
        degraded_img (Tensor): Input degraded image.
        save_path (str): Directory to save images.
        idx (int): Index for filename.
    """
    output_img = output_img.clamp(0, 1).cpu()
    clean_img = clean_img.clamp(0, 1).cpu()
    degraded_img = degraded_img.clamp(0, 1).cpu()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(TF.to_pil_image(degraded_img))
    axs[0].set_title('Degraded')
    axs[1].imshow(TF.to_pil_image(output_img))
    axs[1].set_title('Output')
    axs[2].imshow(TF.to_pil_image(clean_img))
    axs[2].set_title('Ground Truth')

    for ax in axs:
        ax.axis('off')

    os.makedirs(save_path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"comparison_{idx}.png"))
    plt.close()


def evaluate_model(model, val_loader, device, patch_size=128, stride=64, full_size=256,
                   save_visuals=False, save_path='eval_outputs'):
    """
    Evaluate the model using PSNR metric and optionally save visual results.

    Args:
        model (nn.Module): The image restoration model.
        val_loader (DataLoader): Validation data loader.
        device (torch.device): Device to use.
        patch_size (int): Patch size.
        stride (int): Stride used for patches.
        full_size (int): Full image size.
        save_visuals (bool): Whether to save visual comparisons.
        save_path (str): Path to save images.

    Returns:
        float: Average PSNR across validation samples.
    """
    model.eval()
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for batch_idx, (degraded_patches, clean_patches) in enumerate(val_loader):
            degraded_patches = degraded_patches.to(device)
            clean_patches = clean_patches.to(device)

            output_patches = model(degraded_patches)

            output_full = reconstruct_from_patches(
                output_patches, patch_size, stride, full_size)
            clean_full = reconstruct_from_patches(
                clean_patches, patch_size, stride, full_size)
            degraded_full = reconstruct_from_patches(
                degraded_patches, patch_size, stride, full_size)

            mse = torch.mean((output_full - clean_full) ** 2)
            psnr = 10 * torch.log10(1 / mse)

            total_psnr += psnr.item()
            count += 1

            if save_visuals:
                save_comparison_image(
                    output_full, clean_full, degraded_full, save_path, batch_idx)

    avg_psnr = total_psnr / count if count > 0 else 0
    return avg_psnr


def evaluate(args):
    """
    Load dataset, initialize model and run evaluation.

    Args:
        args (argparse.Namespace): Command-line arguments with attributes:
            - save_dir
            - degraded_path
            - clean_path
            - batch_size
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    train_dataset = PromptIRDataset(
        degraded_npz_path=args.degraded_path,
        clean_npz_path=args.clean_path,
        augment=False
    )

    dataset_size = len(train_dataset)
    val_size = int(dataset_size * 0.005)
    train_size = dataset_size - val_size

    _, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    val_loader = DataLoader(
        val_set,
        batch_size=9,  # assumes each validation image is split into 9 patches
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Dataset loaded. Validation samples: {len(val_set)}")

    model = PromptIR_Simplified(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias'
    ).to(device)

    checkpoint_path = 'checkpoints/random.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    model.eval()
    print("Model loaded successfully.")

    val_psnr = evaluate_model(
        model, val_loader, device, save_visuals=True, save_path=args.save_dir)
    print(f"Validation PSNR: {val_psnr:.2f} dB")
