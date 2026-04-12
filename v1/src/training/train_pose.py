"""
WiFi-DensePose training script.

Trains the ModalityTranslationNetwork + DensePoseHead pipeline on
paired CSI + keypoint data (preprocessed by preprocess.py).

Usage:
    python -m src.training.train_pose --data data/processed/ --epochs 50 --output models/
"""

import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def build_models(device: torch.device):
    """Build the modality translator and DensePose head.

    Returns:
        Tuple of (modality_translator, densepose_head).
    """
    from v1.src.models.modality_translation import ModalityTranslationNetwork
    from v1.src.models.densepose_head import DensePoseHead

    mt_config = {
        'input_channels': 64,
        'hidden_channels': [128, 256, 512],
        'output_channels': 256,
        'use_attention': True,
    }
    dp_config = {
        'input_channels': 256,
        'num_body_parts': 24,
        'num_uv_coordinates': 2,
        'hidden_channels': [128, 64],
    }

    mt = ModalityTranslationNetwork(mt_config).to(device)
    dp = DensePoseHead(dp_config).to(device)
    return mt, dp


def keypoints_to_targets(keypoints: torch.Tensor, seg_h: int, seg_w: int):
    """Convert keypoint ground truth to segmentation + UV targets.

    Args:
        keypoints: [B, 17, 3] tensor (x, y, confidence).
        seg_h, seg_w: Spatial dimensions of segmentation output.

    Returns:
        Tuple of (seg_target [B, seg_h, seg_w], uv_target [B, 2, seg_h, seg_w]).
    """
    batch_size = keypoints.shape[0]

    # Create segmentation target: body parts at keypoint locations
    seg_target = torch.zeros(batch_size, seg_h, seg_w, dtype=torch.long)
    uv_target = torch.zeros(batch_size, 2, seg_h, seg_w)

    # Map COCO keypoints to body part IDs (1-indexed, 0 = background)
    kp_to_part = {
        0: 23, 1: 23, 2: 23, 3: 23, 4: 23,  # head
        5: 3, 6: 4,    # shoulders
        7: 5, 8: 6,    # elbows
        9: 7, 10: 8,   # wrists
        11: 1, 12: 2,  # hips
        13: 9, 14: 10, # knees
        15: 11, 16: 12, # ankles
    }

    for b in range(batch_size):
        for kp_idx in range(17):
            x = keypoints[b, kp_idx, 0].item()
            y = keypoints[b, kp_idx, 1].item()
            conf = keypoints[b, kp_idx, 2].item()

            if conf > 0.3:
                px = int(min(max(x * seg_w, 0), seg_w - 1))
                py = int(min(max(y * seg_h, 0), seg_h - 1))
                part_id = kp_to_part.get(kp_idx, 1)
                if part_id < 25:
                    seg_target[b, py, px] = part_id
                uv_target[b, 0, py, px] = x
                uv_target[b, 1, py, px] = y

    return seg_target, uv_target


def train_epoch(mt, dp, dataloader, optimizer, device):
    """Train for one epoch.

    Returns:
        Dict with loss components.
    """
    mt.train()
    dp.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_uv_loss = 0.0
    n_batches = 0

    for csi_batch, kp_batch in dataloader:
        csi_batch = csi_batch.to(device)
        kp_batch = kp_batch.to(device)

        # Forward pass
        visual_features = mt(csi_batch)
        predictions = dp(visual_features)

        # Create targets from keypoints
        seg_h, seg_w = predictions['segmentation'].shape[2:]
        seg_target, uv_target = keypoints_to_targets(kp_batch, seg_h, seg_w)
        seg_target = seg_target.to(device)
        uv_target = uv_target.to(device)

        # Compute losses
        seg_loss = F.cross_entropy(predictions['segmentation'], seg_target, ignore_index=-1)
        uv_loss = F.l1_loss(predictions['uv_coordinates'], uv_target)
        loss = seg_loss + 0.5 * uv_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(mt.parameters()) + list(dp.parameters()), max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_uv_loss += uv_loss.item()
        n_batches += 1

    return {
        'loss': total_loss / max(n_batches, 1),
        'seg_loss': total_seg_loss / max(n_batches, 1),
        'uv_loss': total_uv_loss / max(n_batches, 1),
    }


@torch.no_grad()
def validate(mt, dp, dataloader, device):
    """Validate on a dataset.

    Returns:
        Dict with loss components.
    """
    mt.eval()
    dp.eval()

    total_loss = 0.0
    n_batches = 0

    for csi_batch, kp_batch in dataloader:
        csi_batch = csi_batch.to(device)
        kp_batch = kp_batch.to(device)

        visual_features = mt(csi_batch)
        predictions = dp(visual_features)

        seg_h, seg_w = predictions['segmentation'].shape[2:]
        seg_target, uv_target = keypoints_to_targets(kp_batch, seg_h, seg_w)
        seg_target = seg_target.to(device)
        uv_target = uv_target.to(device)

        seg_loss = F.cross_entropy(predictions['segmentation'], seg_target, ignore_index=-1)
        uv_loss = F.l1_loss(predictions['uv_coordinates'], uv_target)
        loss = seg_loss + 0.5 * uv_loss

        total_loss += loss.item()
        n_batches += 1

    return {'loss': total_loss / max(n_batches, 1)}


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device_name: str = 'cpu',
):
    """Full training loop.

    Args:
        data_dir: Directory with train/val/test subdirs containing .pt files.
        output_dir: Directory to save model checkpoints.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        device_name: 'cpu' or 'cuda'.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name)

    # Load data
    train_csi = torch.load(os.path.join(data_dir, 'train', 'csi.pt'), weights_only=True)
    train_kp = torch.load(os.path.join(data_dir, 'train', 'keypoints.pt'), weights_only=True)
    val_csi = torch.load(os.path.join(data_dir, 'val', 'csi.pt'), weights_only=True)
    val_kp = torch.load(os.path.join(data_dir, 'val', 'keypoints.pt'), weights_only=True)

    logger.info(f"Train: {len(train_csi)} samples, Val: {len(val_csi)} samples")

    train_loader = DataLoader(
        TensorDataset(train_csi, train_kp), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_csi, val_kp), batch_size=batch_size
    )

    # Build models
    mt, dp = build_models(device)
    total_params = sum(p.numel() for p in mt.parameters()) + sum(p.numel() for p in dp.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(mt.parameters()) + list(dp.parameters()),
        lr=learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_metrics = train_epoch(mt, dp, train_loader, optimizer, device)
        val_metrics = validate(mt, dp, val_loader, device)
        scheduler.step()
        elapsed = time.time() - start

        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train loss: {train_metrics['loss']:.4f} "
            f"(seg: {train_metrics['seg_loss']:.4f}, uv: {train_metrics['uv_loss']:.4f}) | "
            f"Val loss: {val_metrics['loss']:.4f}"
        )

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch,
                'mt_state_dict': mt.state_dict(),
                'dp_state_dict': dp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, os.path.join(output_dir, 'best.pt'))
            logger.info(f"  New best model saved (val_loss={best_val_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': epochs,
        'mt_state_dict': mt.state_dict(),
        'dp_state_dict': dp.state_dict(),
        'val_loss': val_metrics['loss'],
    }, os.path.join(output_dir, 'final.pt'))
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train WiFi-DensePose model')
    parser.add_argument('--data', required=True, help='Preprocessed data directory')
    parser.add_argument('--output', default='models/', help='Output directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    train(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_name=args.device,
    )


if __name__ == '__main__':
    main()
