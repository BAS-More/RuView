"""
Room calibration script for WiFi-DensePose.

Records baseline CSI data in an empty room to establish the noise floor
and environmental characteristics for the sensing system.

Usage:
    python scripts/calibrate_room.py --duration 30
    python scripts/calibrate_room.py --mock --duration 10
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np

logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'v1'))


def calibrate_mock(output_path: str, duration: float):
    """Mock calibration using synthetic CSI data."""
    from src.testing.mock_csi_generator import MockCSIGenerator

    gen = MockCSIGenerator(
        num_subcarriers=64, num_antennas=3, num_samples=100,
        noise_level=0.05,  # Low noise for baseline
        movement_amplitude=0.0,  # No movement
    )

    print("Recording empty room baseline (mock mode)...")
    frames = []
    start = time.time()

    while time.time() - start < duration:
        csi = gen.generate()
        amplitude = np.abs(csi)
        frames.append({
            'timestamp': time.time(),
            'amplitude_mean': float(amplitude.mean()),
            'amplitude_std': float(amplitude.std()),
            'amplitude_max': float(amplitude.max()),
            'phase_std': float(np.std(np.angle(csi))),
        })
        time.sleep(0.05)  # 20 Hz

    if not frames:
        print("ERROR: No frames collected!")
        return

    # Compute calibration parameters
    amp_means = [f['amplitude_mean'] for f in frames]
    amp_stds = [f['amplitude_std'] for f in frames]
    phase_stds = [f['phase_std'] for f in frames]

    calibration = {
        'timestamp': time.time(),
        'duration_seconds': duration,
        'n_frames': len(frames),
        'noise_floor': {
            'amplitude_mean': float(np.mean(amp_means)),
            'amplitude_std': float(np.mean(amp_stds)),
            'amplitude_variance_threshold': float(np.mean(amp_stds) * 3),
            'phase_std': float(np.mean(phase_stds)),
        },
        'environment': {
            'baseline_rssi': -45.0,
            'signal_stability': float(1.0 - np.std(amp_means) / np.mean(amp_means)),
        },
        'thresholds': {
            'presence_variance': float(np.mean(amp_stds) * 5),
            'motion_energy': float(np.mean(amp_stds) * 10),
        },
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"\nCalibration complete!")
    print(f"  Frames recorded: {len(frames)}")
    print(f"  Noise floor amplitude: {calibration['noise_floor']['amplitude_mean']:.4f} "
          f"(±{calibration['noise_floor']['amplitude_std']:.4f})")
    print(f"  Presence threshold: {calibration['thresholds']['presence_variance']:.4f}")
    print(f"  Motion threshold: {calibration['thresholds']['motion_energy']:.4f}")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Calibrate room for WiFi-DensePose')
    parser.add_argument('--duration', type=float, default=30, help='Calibration duration in seconds')
    parser.add_argument('--output', default='data/calibration.json', help='Output calibration file')
    parser.add_argument('--mock', action='store_true', help='Use mock data')
    parser.add_argument('--mode', default='baseline', choices=['baseline', 'walking'],
                        help='Calibration mode')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    print(f"RuView Room Calibration")
    print(f"  Mode: {args.mode} ({'mock' if args.mock else 'live'})")
    print(f"  Duration: {args.duration}s")
    print()

    if args.mock:
        calibrate_mock(args.output, args.duration)
    else:
        print("Live calibration requires ESP32 sensors streaming CSI data.")
        print("See docs/HOME_SETUP.md for hardware setup instructions.")
        print("Use --mock flag to test without hardware.")


if __name__ == '__main__':
    main()
