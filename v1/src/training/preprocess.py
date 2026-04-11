"""
Data preprocessing pipeline for WiFi-DensePose training.

Loads raw CSI JSONL recordings, extracts amplitude + phase features,
and optionally pairs with camera-based keypoint ground truth.

Usage:
    python -m src.training.preprocess --input data/recordings/ --output data/processed/
"""

import argparse
import json
import logging
import os
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def parse_iq_hex(iq_hex: str, n_subcarriers: int) -> Tuple[np.ndarray, np.ndarray]:
    """Parse IQ hex string into amplitude and phase arrays.

    The ESP32 CSI firmware encodes I/Q as signed bytes in hex.

    Args:
        iq_hex: Hex-encoded I/Q pairs (2 bytes per subcarrier).
        n_subcarriers: Expected number of subcarriers.

    Returns:
        Tuple of (amplitude, phase) arrays, each shape (n_subcarriers,).
    """
    raw = bytes.fromhex(iq_hex)
    n_pairs = len(raw) // 2
    actual_sc = min(n_pairs, n_subcarriers)

    i_vals = np.zeros(actual_sc, dtype=np.float32)
    q_vals = np.zeros(actual_sc, dtype=np.float32)

    for idx in range(actual_sc):
        # Signed bytes: I then Q
        i_val = raw[idx * 2]
        q_val = raw[idx * 2 + 1]
        # Convert unsigned to signed
        if i_val > 127:
            i_val -= 256
        if q_val > 127:
            q_val -= 256
        i_vals[idx] = float(i_val)
        q_vals[idx] = float(q_val)

    amplitude = np.sqrt(i_vals ** 2 + q_vals ** 2)
    phase = np.arctan2(q_vals, i_vals)
    return amplitude, phase


def load_csi_jsonl(filepath: str, max_frames: Optional[int] = None) -> List[Dict]:
    """Load CSI frames from a JSONL file.

    Args:
        filepath: Path to .csi.jsonl file.
        max_frames: Maximum number of frames to load (None = all).

    Returns:
        List of parsed frame dicts with 'timestamp', 'amplitude', 'phase', 'rssi'.
    """
    frames = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_frames and i >= max_frames:
                break
            try:
                raw = json.loads(line.strip())
                n_sc = raw.get('subcarriers', 64)
                amp, phase = parse_iq_hex(raw['iq_hex'], n_sc)
                frames.append({
                    'timestamp': raw['timestamp'],
                    'amplitude': amp,
                    'phase': phase,
                    'rssi': raw.get('rssi', 0),
                    'node_id': raw.get('node_id', 'unknown'),
                    'n_subcarriers': len(amp),
                })
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed frame {i}: {e}")
                continue
    return frames


def frames_to_tensor(
    frames: List[Dict],
    target_channels: int = 64,
    spatial_size: int = 8,
) -> torch.Tensor:
    """Convert parsed CSI frames into a batched tensor.

    Each frame's amplitude + phase are concatenated and reshaped to
    [channels, H, W] suitable for the ModalityTranslationNetwork.

    Args:
        frames: List of parsed CSI frame dicts.
        target_channels: Number of channels (default 64).
        spatial_size: H and W spatial dimensions (default 8).

    Returns:
        Tensor of shape [N, target_channels, spatial_size, spatial_size].
    """
    needed = target_channels * spatial_size * spatial_size
    tensors = []

    for frame in frames:
        # Concatenate amplitude and phase
        features = np.concatenate([frame['amplitude'], frame['phase']])
        # Pad or truncate
        if len(features) < needed:
            features = np.pad(features, (0, needed - len(features)))
        else:
            features = features[:needed]
        tensor = torch.from_numpy(features).float().view(
            target_channels, spatial_size, spatial_size
        )
        tensors.append(tensor)

    if not tensors:
        return torch.zeros(0, target_channels, spatial_size, spatial_size)
    return torch.stack(tensors)


def load_keypoints_jsonl(filepath: str, max_frames: Optional[int] = None) -> List[Dict]:
    """Load MediaPipe keypoint ground-truth from JSONL.

    Args:
        filepath: Path to keypoints JSONL file.
        max_frames: Max frames to load.

    Returns:
        List of dicts with 'timestamp' and 'keypoints' (17x3 array).
    """
    frames = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_frames and i >= max_frames:
                break
            try:
                raw = json.loads(line.strip())
                frames.append({
                    'timestamp': raw['timestamp'],
                    'keypoints': np.array(raw['keypoints'], dtype=np.float32),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed keypoint frame {i}: {e}")
    return frames


def create_synthetic_ground_truth(n_frames: int) -> torch.Tensor:
    """Create synthetic keypoint ground truth for dry-run testing.

    Generates random but plausible 17-keypoint COCO poses.

    Args:
        n_frames: Number of frames.

    Returns:
        Tensor of shape [n_frames, 17, 3] (x, y, confidence).
    """
    keypoints = torch.zeros(n_frames, 17, 3)
    for i in range(n_frames):
        # Random body center
        cx, cy = 0.5 + 0.2 * torch.randn(1).item(), 0.5 + 0.2 * torch.randn(1).item()
        for j in range(17):
            keypoints[i, j, 0] = max(0, min(1, cx + 0.1 * torch.randn(1).item()))
            keypoints[i, j, 1] = max(0, min(1, cy + 0.15 * torch.randn(1).item()))
            keypoints[i, j, 2] = 0.5 + 0.3 * torch.rand(1).item()  # confidence
    return keypoints


def preprocess_dataset(
    csi_dir: str,
    output_dir: str,
    max_frames: Optional[int] = None,
    keypoints_file: Optional[str] = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    """Full preprocessing pipeline: load CSI, optionally pair with keypoints, split, save.

    Args:
        csi_dir: Directory containing .csi.jsonl files.
        output_dir: Directory to save processed tensors.
        max_frames: Max frames per file.
        keypoints_file: Optional keypoints JSONL for paired dataset.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all CSI files
    all_frames = []
    csi_path = Path(csi_dir)
    for jsonl_file in sorted(csi_path.glob('*.csi.jsonl')):
        logger.info(f"Loading {jsonl_file.name}...")
        frames = load_csi_jsonl(str(jsonl_file), max_frames=max_frames)
        all_frames.extend(frames)
        logger.info(f"  Loaded {len(frames)} frames")

    if not all_frames:
        logger.error("No CSI frames found!")
        return

    logger.info(f"Total frames: {len(all_frames)}")

    # Convert to tensors
    csi_tensor = frames_to_tensor(all_frames)
    logger.info(f"CSI tensor shape: {csi_tensor.shape}")

    # Ground truth (real or synthetic)
    if keypoints_file and os.path.exists(keypoints_file):
        kp_frames = load_keypoints_jsonl(keypoints_file, max_frames=len(all_frames))
        # TODO: Timestamp alignment between CSI and keypoints
        gt_tensor = torch.tensor(
            [f['keypoints'] for f in kp_frames[:len(all_frames)]],
            dtype=torch.float32,
        )
    else:
        logger.warning("No keypoints file — generating synthetic ground truth for dry-run")
        gt_tensor = create_synthetic_ground_truth(len(all_frames))

    # Ensure same length
    n = min(len(csi_tensor), len(gt_tensor))
    csi_tensor = csi_tensor[:n]
    gt_tensor = gt_tensor[:n]

    # Split
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        'train': (csi_tensor[:n_train], gt_tensor[:n_train]),
        'val': (csi_tensor[n_train:n_train + n_val], gt_tensor[n_train:n_train + n_val]),
        'test': (csi_tensor[n_train + n_val:], gt_tensor[n_train + n_val:]),
    }

    for name, (csi, gt) in splits.items():
        split_dir = os.path.join(output_dir, name)
        os.makedirs(split_dir, exist_ok=True)
        torch.save(csi, os.path.join(split_dir, 'csi.pt'))
        torch.save(gt, os.path.join(split_dir, 'keypoints.pt'))
        logger.info(f"  {name}: {len(csi)} frames saved")

    logger.info(f"Preprocessing complete. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Preprocess CSI data for training')
    parser.add_argument('--input', required=True, help='Directory with .csi.jsonl files')
    parser.add_argument('--output', required=True, help='Output directory for processed tensors')
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames per file')
    parser.add_argument('--keypoints', default=None, help='Keypoints JSONL for ground truth')
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    preprocess_dataset(
        csi_dir=args.input,
        output_dir=args.output,
        max_frames=args.max_frames,
        keypoints_file=args.keypoints,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


if __name__ == '__main__':
    main()
