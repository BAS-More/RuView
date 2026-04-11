"""
Evaluation script for WiFi-DensePose models.

Computes PCK@0.2 (Percentage of Correct Keypoints), per-joint accuracy,
and inference latency.

Usage:
    python -m src.training.evaluate --model models/best.pt --data data/processed/test/
"""

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


def compute_pck(pred_kp: np.ndarray, gt_kp: np.ndarray, threshold: float = 0.2) -> dict:
    """Compute PCK (Percentage of Correct Keypoints).

    Args:
        pred_kp: Predicted keypoints [N, 17, 2] (x, y in [0,1]).
        gt_kp: Ground truth keypoints [N, 17, 2].
        threshold: Distance threshold as fraction of torso size.

    Returns:
        Dict with overall PCK and per-joint PCK.
    """
    keypoint_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    n_samples, n_kp = pred_kp.shape[0], pred_kp.shape[1]

    # Torso size: distance between left_shoulder (5) and left_hip (11)
    torso_sizes = np.linalg.norm(gt_kp[:, 5, :2] - gt_kp[:, 11, :2], axis=1)
    torso_sizes = np.maximum(torso_sizes, 0.05)  # minimum to avoid division by zero

    # Per-sample, per-keypoint distances
    distances = np.linalg.norm(pred_kp[:, :, :2] - gt_kp[:, :, :2], axis=2)  # [N, 17]

    # Normalize by torso size
    normalized = distances / torso_sizes[:, None]

    # PCK: fraction within threshold
    correct = normalized < threshold
    overall_pck = correct.mean()

    per_joint = {}
    for j in range(min(n_kp, 17)):
        per_joint[keypoint_names[j]] = correct[:, j].mean()

    return {
        'pck': float(overall_pck),
        'per_joint': per_joint,
        'mean_distance': float(distances.mean()),
        'median_distance': float(np.median(distances)),
    }


def extract_keypoints_from_output(seg_probs: torch.Tensor, uv: torch.Tensor) -> np.ndarray:
    """Extract keypoint locations from DensePose segmentation output.

    Args:
        seg_probs: Softmax segmentation [B, 25, H, W].
        uv: UV coordinates [B, 2, H, W].

    Returns:
        Keypoints array [B, 17, 2] (x, y in [0,1]).
    """
    batch_size = seg_probs.shape[0]
    _, _, h, w = seg_probs.shape

    part_to_kp = {
        0: 23, 1: 23, 2: 23, 3: 23, 4: 23,
        5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
        11: 1, 12: 2, 13: 9, 14: 10, 15: 11, 16: 12,
    }

    keypoints = np.zeros((batch_size, 17, 2), dtype=np.float32)

    for b in range(batch_size):
        for kp_idx in range(17):
            part_idx = part_to_kp.get(kp_idx, 1)
            if part_idx < seg_probs.shape[1]:
                part_map = seg_probs[b, part_idx].cpu().numpy()
                flat_idx = part_map.argmax()
                y_px = flat_idx // w
                x_px = flat_idx % w
                keypoints[b, kp_idx, 0] = x_px / max(w - 1, 1)
                keypoints[b, kp_idx, 1] = y_px / max(h - 1, 1)

    return keypoints


def evaluate(
    model_path: str,
    data_dir: str,
    batch_size: int = 16,
    device_name: str = 'cpu',
):
    """Evaluate a trained model on test data.

    Args:
        model_path: Path to saved model checkpoint.
        data_dir: Directory with csi.pt and keypoints.pt.
        batch_size: Batch size for inference.
        device_name: 'cpu' or 'cuda'.
    """
    device = torch.device(device_name)

    # Load model
    from src.training.train_pose import build_models
    mt, dp = build_models(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    mt.load_state_dict(checkpoint['mt_state_dict'])
    dp.load_state_dict(checkpoint['dp_state_dict'])
    mt.eval()
    dp.eval()
    logger.info(f"Model loaded from {model_path} (epoch {checkpoint.get('epoch', '?')})")

    # Load test data
    test_csi = torch.load(os.path.join(data_dir, 'csi.pt'), weights_only=True)
    test_kp = torch.load(os.path.join(data_dir, 'keypoints.pt'), weights_only=True)
    logger.info(f"Test data: {len(test_csi)} samples")

    loader = DataLoader(TensorDataset(test_csi, test_kp), batch_size=batch_size)

    all_pred_kp = []
    all_gt_kp = []
    latencies = []

    with torch.no_grad():
        for csi_batch, kp_batch in loader:
            csi_batch = csi_batch.to(device)

            start = time.perf_counter()
            visual = mt(csi_batch)
            output = dp(visual)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed / len(csi_batch))

            seg_probs = torch.softmax(output['segmentation'], dim=1)
            pred_kp = extract_keypoints_from_output(seg_probs, output['uv_coordinates'])
            all_pred_kp.append(pred_kp)
            all_gt_kp.append(kp_batch.numpy())

    pred_kp = np.concatenate(all_pred_kp)
    gt_kp = np.concatenate(all_gt_kp)

    # Compute metrics
    pck_results = compute_pck(pred_kp, gt_kp, threshold=0.2)

    avg_latency_ms = np.mean(latencies) * 1000
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0

    # Report
    print("\n" + "=" * 60)
    print("WiFi-DensePose Evaluation Report")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Test samples: {len(pred_kp)}")
    print(f"\nPCK@0.2: {pck_results['pck']:.4f} ({pck_results['pck']*100:.1f}%)")
    print(f"Mean distance: {pck_results['mean_distance']:.4f}")
    print(f"Median distance: {pck_results['median_distance']:.4f}")
    print(f"\nPer-joint PCK@0.2:")
    for name, val in pck_results['per_joint'].items():
        bar = '#' * int(val * 40)
        print(f"  {name:20s} {val:.3f} {bar}")
    print(f"\nInference latency: {avg_latency_ms:.1f} ms/frame ({fps:.0f} FPS)")
    print("=" * 60)

    return pck_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate WiFi-DensePose model')
    parser.add_argument('--model', required=True, help='Path to model checkpoint')
    parser.add_argument('--data', required=True, help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    evaluate(args.model, args.data, args.batch_size, args.device)


if __name__ == '__main__':
    main()
