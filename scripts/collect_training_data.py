"""
Simultaneous CSI + camera data collection for WiFi-DensePose training.

Records CSI frames from ESP32 sensors and webcam keypoints (via MediaPipe)
simultaneously, with timestamp alignment for paired training data.

Usage:
    python scripts/collect_training_data.py --duration 30m --output data/recordings/session1/

Mock mode (no hardware):
    python scripts/collect_training_data.py --mock --duration 10s --output data/recordings/mock/
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Add v1 to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'v1'))


def parse_duration(s: str) -> float:
    """Parse duration string like '30m', '1h', '10s' to seconds."""
    s = s.strip().lower()
    if s.endswith('h'):
        return float(s[:-1]) * 3600
    elif s.endswith('m'):
        return float(s[:-1]) * 60
    elif s.endswith('s'):
        return float(s[:-1])
    return float(s)


def collect_mock(output_dir: str, duration: float):
    """Mock data collection without hardware.

    Generates synthetic CSI and keypoint data for testing the pipeline.
    """
    from src.testing.mock_csi_generator import MockCSIGenerator

    os.makedirs(output_dir, exist_ok=True)
    csi_path = os.path.join(output_dir, 'mock.csi.jsonl')
    kp_path = os.path.join(output_dir, 'keypoints.jsonl')

    gen = MockCSIGenerator(num_subcarriers=64, num_antennas=3, num_samples=100)

    start = time.time()
    frame_count = 0
    target_fps = 20  # 20 Hz

    with open(csi_path, 'w') as csi_f, open(kp_path, 'w') as kp_f:
        while time.time() - start < duration:
            ts = time.time()

            # Generate mock CSI
            csi = gen.generate()
            amplitude = np.abs(csi[0, :, 0]).tolist()  # first antenna, first sample
            iq_hex = ''.join(f'{int(a) & 0xFF:02x}{int(p) & 0xFF:02x}'
                             for a, p in zip(amplitude[:64], amplitude[:64]))

            csi_frame = {
                'timestamp': ts,
                'node_id': 'mock-sensor-1',
                'magic': 0,
                'size': 128,
                'rssi': -45 + int(5 * np.random.randn()),
                'type': 0,
                'iq_hex': iq_hex,
                'subcarriers': 64,
            }
            csi_f.write(json.dumps(csi_frame) + '\n')

            # Generate mock keypoints (random walk)
            cx = 0.5 + 0.2 * np.sin(ts * 0.5)
            cy = 0.5 + 0.15 * np.cos(ts * 0.3)
            keypoints = []
            for j in range(17):
                keypoints.append([
                    float(max(0, min(1, cx + 0.08 * np.random.randn()))),
                    float(max(0, min(1, cy + 0.1 * np.random.randn()))),
                    float(0.5 + 0.3 * np.random.rand()),
                ])
            kp_frame = {'timestamp': ts, 'keypoints': keypoints}
            kp_f.write(json.dumps(kp_frame) + '\n')

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start
                fps = frame_count / elapsed
                print(f'\r  Frames: {frame_count} | FPS: {fps:.1f} | '
                      f'Elapsed: {elapsed:.0f}s / {duration:.0f}s', end='', flush=True)

            # Rate limit to target FPS
            sleep_time = (1.0 / target_fps) - (time.time() - ts)
            if sleep_time > 0:
                time.sleep(sleep_time)

    print(f'\n  Done! {frame_count} frames saved to {output_dir}')
    print(f'  CSI: {csi_path}')
    print(f'  Keypoints: {kp_path}')


def collect_live(output_dir: str, duration: float, camera_idx: int = 0):
    """Live data collection with real ESP32 sensors and webcam.

    Requires:
    - ESP32 sensors streaming CSI via UDP
    - Webcam connected
    - MediaPipe installed (pip install mediapipe)
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("ERROR: Live collection requires opencv-python and mediapipe.")
        print("Install: pip install opencv-python mediapipe")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
    )

    # Open webcam
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_idx}")
        sys.exit(1)

    kp_path = os.path.join(output_dir, 'keypoints.jsonl')
    # CSI data would come from UDP listener (not implemented in this script)
    # For now, just collect camera keypoints

    print(f"Camera opened (index {camera_idx}). Press Ctrl+C to stop.")
    print("NOTE: CSI collection requires ESP32 sensors running. See docs/HOME_SETUP.md")

    start = time.time()
    frame_count = 0

    with open(kp_path, 'w') as kp_f:
        try:
            while time.time() - start < duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Process with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    keypoints = []
                    for lm in results.pose_landmarks.landmark[:17]:
                        keypoints.append([lm.x, lm.y, lm.visibility])

                    kp_frame = {
                        'timestamp': time.time(),
                        'keypoints': keypoints,
                    }
                    kp_f.write(json.dumps(kp_frame) + '\n')
                    frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - start
                    print(f'\r  Keypoint frames: {frame_count} | '
                          f'Elapsed: {elapsed:.0f}s / {duration:.0f}s', end='', flush=True)

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cap.release()
            pose.close()

    print(f'\n  Done! {frame_count} keypoint frames saved to {kp_path}')


def main():
    parser = argparse.ArgumentParser(description='Collect training data for WiFi-DensePose')
    parser.add_argument('--duration', default='30m', help='Collection duration (e.g. 30m, 1h, 10s)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--camera', type=int, default=0, help='Camera index for live collection')
    parser.add_argument('--mock', action='store_true', help='Use mock data (no hardware needed)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    duration = parse_duration(args.duration)

    print(f"RuView Data Collection")
    print(f"  Duration: {duration:.0f}s ({args.duration})")
    print(f"  Output: {args.output}")
    print(f"  Mode: {'mock' if args.mock else 'live'}")
    print()

    if args.mock:
        collect_mock(args.output, duration)
    else:
        collect_live(args.output, duration, args.camera)


if __name__ == '__main__':
    main()
