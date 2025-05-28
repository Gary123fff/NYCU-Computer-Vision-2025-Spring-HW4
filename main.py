"""
Main script to train, evaluate, or test an image restoration model.
"""

import argparse

from test import run_test_inference
from train import train_with_metrics
from process import process_and_show_images, process_test_degraded_images
from evaluate import evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--degraded_path', type=str,
                        default='output_npz_patches/train_degraded_patches.npz')
    parser.add_argument('--clean_path', type=str,
                        default='output_npz_patches/train_clean_patches.npz')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    args = parser.parse_args()

    if args.mode == 'train':
        process_and_show_images(sample_num=0)
        history = train_with_metrics(args)
        print(f"Training completed. Best PSNR: {history['best_psnr']:.2f}dB")

    elif args.mode == 'val':
        evaluate(args)

    elif args.mode == 'test':
        process_test_degraded_images(base_path='hw4_realse_dataset/test')
        run_test_inference(
            degraded_npz_path='output_npz_patches/test_degraded_patches.npz',
            model_path='checkpoints/random.pth',
            save_dir='test_outputs',
            num_images=5
        )
