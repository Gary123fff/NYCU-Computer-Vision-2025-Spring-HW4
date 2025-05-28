"""
This module contains evaluation utilities for calculating PSNR,
evaluating models, and plotting training/validation metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import PromptIR_Simplified
from loaddata import PromptIRDataset
import torch


def calculate_psnr(img1, img2):
    """
    Calculate PSNR between two images.
    Args:
        img1: First image tensor [B, C, H, W], values in range [0, 1]
        img2: Second image tensor [B, C, H, W], values in range [0, 1]
    Returns:
        Average PSNR value across the batch
    """
    img1 = img1 * 255.0
    img2 = img2 * 255.0

    mse = ((img1 - img2) ** 2).mean(dim=[1, 2, 3])
    psnr = 20 * torch.log10(torch.tensor(255.0).to(img1.device)
                            ) - 10 * torch.log10(mse)

    return psnr.mean().item()


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on validation set to get PSNR.
    Args:
        model: Neural network model
        data_loader: DataLoader for validation data
        device: Device to run evaluation on
    Returns:
        Average PSNR value
    """
    model.eval()
    psnr_values = []

    with torch.no_grad():
        for degraded, clean in tqdm(data_loader, desc="Evaluating"):
            degraded = degraded.to(device)
            clean = clean.to(device)

            output = model(degraded)
            psnr = calculate_psnr(output, clean)
            psnr_values.append(psnr)

    return np.mean(psnr_values)


def plot_metrics(train_losses, val_psnrs, epoch, final=False):
    """
    Plot training loss and validation PSNR.
    Args:
        train_losses: List of training losses
        val_psnrs: List of validation PSNR values
        epoch: Current epoch number
        final: Whether this is the final plot
    """
    plt.figure(figsize=(12, 10))

    # Plot Loss
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1),
             train_losses, 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot PSNR
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(val_psnrs) + 1), val_psnrs,
             'r-', label='Validation PSNR')
    plt.title('Validation PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    filename = f"metrics_final.png" if final else f"metrics_epoch_{epoch}.png"
    plt.savefig(filename)
    plt.close()


def evaluate_checkpoint(checkpoint_path, data_path, batch_size=4):
    """
    Evaluate a saved model checkpoint on test data.
    Args:
        checkpoint_path: Path to the model checkpoint
        data_path: Dictionary with paths to 'degraded' and 'clean' data
        batch_size: Batch size for evaluation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded model from {checkpoint_path}")

    test_dataset = PromptIRDataset(
        degraded_npz_path=data_path['degraded'],
        clean_npz_path=data_path['clean']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Test dataset loaded with {len(test_dataset)} samples")

    psnr = evaluate_model(model, test_loader, device)
    print(f"Test PSNR: {psnr:.2f} dB")

    return psnr
