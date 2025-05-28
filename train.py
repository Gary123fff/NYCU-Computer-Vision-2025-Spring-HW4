"""Train script for PromptIR with perceptual loss and metrics tracking."""

import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
from tqdm import tqdm

from loaddata import PromptIRDataset
from model import PromptIR_Simplified
from metrics import plot_metrics
from evaluate import evaluate_model


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss for image restoration tasks."""

    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        """Compute perceptual L1 loss using VGG features."""
        pred = (pred.float() - self.mean) / self.std
        target = (target.float() - self.mean) / self.std
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def train_with_metrics(args):
    """Main training loop with PSNR evaluation and checkpointing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    train_dataset = PromptIRDataset(
        degraded_npz_path=args.degraded_path,
        clean_npz_path=args.clean_path,
        augment=True
    )

    dataset_size = len(train_dataset)
    val_size = int(dataset_size * 0.005)
    train_size = dataset_size - val_size

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_patch_num = 9
    val_loader = DataLoader(
        val_set,
        batch_size=val_patch_num,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(
        f"Dataset loaded. Training samples: {len(train_set)}, Validation samples: {len(val_set)}")

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

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {param_count:,} parameters")

    l2_loss = nn.MSELoss()
    perceptual_loss = VGGPerceptualLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_losses = []
    val_psnrs = []
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for degraded, clean in pbar:
            degraded = degraded.float().to(device)
            clean = clean.float().to(device)

            optimizer.zero_grad()
            output = model(degraded)

            loss = l2_loss(output, clean) + 0.01 * \
                perceptual_loss(output, clean)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            pbar.set_postfix(loss=current_loss)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        val_psnr = evaluate_model(model, val_loader, device)
        val_psnrs.append(val_psnr)

        print(f"Epoch [{epoch}/{args.epochs}] - Avg Loss: {avg_loss:.4f} - Val PSNR: {val_psnr:.2f}dB - "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_model_path = os.path.join(args.save_dir, "random.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'psnr': val_psnr,
            }, best_model_path)
            print(
                f"Best model saved at {best_model_path} (PSNR: {best_psnr:.2f}dB)")

        if epoch % args.save_every == 0 or epoch == args.epochs:
            plot_metrics(train_losses, val_psnrs, epoch)

    plot_metrics(train_losses, val_psnrs, args.epochs, final=True)

    return {
        'train_losses': train_losses,
        'val_psnrs': val_psnrs,
        'best_psnr': best_psnr
    }
