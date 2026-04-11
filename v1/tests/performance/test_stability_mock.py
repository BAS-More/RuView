"""
Long-running stability test — runs 1000 mock frames through the pipeline
and checks for memory growth, crashes, and output consistency.

Simulates physical tests:
  #8  Signal stability (1 hour)
  #20 24-hour run
"""

import sys
import os
import gc
import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.densepose_head import DensePoseHead
from src.models.modality_translation import ModalityTranslationNetwork
from src.testing.mock_csi_generator import MockCSIGenerator


def get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    except ImportError:
        # Fallback: use torch memory tracking
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0


def csi_to_tensor(csi_complex: np.ndarray) -> torch.Tensor:
    amplitude = np.abs(csi_complex).flatten().astype(np.float32)
    n_ch = 64
    h = w = max(1, int((len(amplitude) // n_ch) ** 0.5))
    needed = n_ch * h * w
    if len(amplitude) < needed:
        amplitude = np.pad(amplitude, (0, needed - len(amplitude)))
    else:
        amplitude = amplitude[:needed]
    return torch.from_numpy(amplitude).view(1, n_ch, h, w)


class TestPipelineStability:
    """Stability tests for the inference pipeline under sustained load."""

    @pytest.fixture
    def pipeline(self):
        dp_config = {
            'input_channels': 256, 'num_body_parts': 24,
            'num_uv_coordinates': 2, 'hidden_channels': [128, 64],
        }
        mt_config = {
            'input_channels': 64, 'hidden_channels': [128, 256, 512],
            'output_channels': 256, 'use_attention': True,
        }
        dp = DensePoseHead(dp_config)
        mt = ModalityTranslationNetwork(mt_config)
        dp.eval()
        mt.eval()
        gen = MockCSIGenerator(num_subcarriers=64, num_antennas=3, num_samples=100)
        return mt, dp, gen

    def test_1000_frames_no_crash(self, pipeline):
        """Run 1000 frames without any crash or exception."""
        mt, dp, gen = pipeline
        for i in range(1000):
            csi = gen.generate()
            tensor = csi_to_tensor(csi)
            with torch.no_grad():
                visual = mt(tensor)
                poses = dp(visual)
            assert 'segmentation' in poses, f"Frame {i} missing segmentation"

    def test_output_consistency(self, pipeline):
        """Output structure stays consistent across all frames."""
        mt, dp, gen = pipeline
        seg_shapes = set()
        uv_shapes = set()
        for _ in range(100):
            csi = gen.generate()
            tensor = csi_to_tensor(csi)
            with torch.no_grad():
                poses = dp(mt(tensor))
            seg_shapes.add(poses['segmentation'].shape)
            uv_shapes.add(poses['uv_coordinates'].shape)
        # All frames should produce same output shapes
        assert len(seg_shapes) == 1, f"Inconsistent seg shapes: {seg_shapes}"
        assert len(uv_shapes) == 1, f"Inconsistent UV shapes: {uv_shapes}"

    def test_no_nan_or_inf(self, pipeline):
        """No NaN or Inf values across 500 frames."""
        mt, dp, gen = pipeline
        for i in range(500):
            csi = gen.generate()
            tensor = csi_to_tensor(csi)
            with torch.no_grad():
                visual = mt(tensor)
                poses = dp(visual)
            assert not torch.isnan(poses['segmentation']).any(), f"NaN in seg at frame {i}"
            assert not torch.isinf(poses['segmentation']).any(), f"Inf in seg at frame {i}"
            assert not torch.isnan(poses['uv_coordinates']).any(), f"NaN in UV at frame {i}"

    def test_memory_does_not_grow(self, pipeline):
        """Memory usage doesn't grow significantly over 500 frames."""
        mt, dp, gen = pipeline
        gc.collect()
        # Warmup
        for _ in range(10):
            csi = gen.generate()
            with torch.no_grad():
                dp(mt(csi_to_tensor(csi)))
        gc.collect()

        baseline_mem = get_memory_mb()
        for _ in range(500):
            csi = gen.generate()
            with torch.no_grad():
                dp(mt(csi_to_tensor(csi)))

        gc.collect()
        final_mem = get_memory_mb()

        if baseline_mem > 0:  # psutil available
            growth = final_mem - baseline_mem
            assert growth < 100, f"Memory grew {growth:.1f} MB over 500 frames"

    def test_throughput_benchmark(self, pipeline):
        """Measure frames per second throughput."""
        import time
        mt, dp, gen = pipeline
        n_frames = 100

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                dp(mt(csi_to_tensor(gen.generate())))

        start = time.perf_counter()
        for _ in range(n_frames):
            csi = gen.generate()
            with torch.no_grad():
                dp(mt(csi_to_tensor(csi)))
        elapsed = time.perf_counter() - start

        fps = n_frames / elapsed
        print(f"\nThroughput: {fps:.1f} FPS ({elapsed:.2f}s for {n_frames} frames)")
        assert fps > 1.0, f"Pipeline too slow: {fps:.1f} FPS"
