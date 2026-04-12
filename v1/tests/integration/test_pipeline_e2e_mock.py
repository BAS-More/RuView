"""
End-to-end pipeline mock tests — validates the entire inference path
from synthetic CSI data through to parsed pose output, with no hardware.

Simulates physical tests:
  #3  CSI data streaming
  #15 Live inference test
"""

import sys
import os
import pytest
import numpy as np
import torch
import torch.nn.functional as F

# Ensure v1/src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from v1.src.models.densepose_head import DensePoseHead
from v1.src.models.modality_translation import ModalityTranslationNetwork
from v1.src.testing.mock_csi_generator import MockCSIGenerator


# --- Fixtures ---

@pytest.fixture
def densepose_config():
    return {
        'input_channels': 256,
        'num_body_parts': 24,
        'num_uv_coordinates': 2,
        'hidden_channels': [128, 64],
    }


@pytest.fixture
def modality_config():
    return {
        'input_channels': 64,
        'hidden_channels': [128, 256, 512],
        'output_channels': 256,
        'use_attention': True,
    }


@pytest.fixture
def densepose_model(densepose_config):
    model = DensePoseHead(densepose_config)
    model.eval()
    return model


@pytest.fixture
def modality_translator(modality_config):
    model = ModalityTranslationNetwork(modality_config)
    model.eval()
    return model


@pytest.fixture
def mock_csi_gen():
    return MockCSIGenerator(
        num_subcarriers=64,
        num_antennas=3,
        num_samples=100,
        noise_level=0.1,
        movement_freq=0.5,
        movement_amplitude=0.3,
    )


# --- Helper: reshape CSI to model input ---

def csi_to_tensor(csi_complex: np.ndarray) -> torch.Tensor:
    """Convert complex CSI array to [1, 64, H, W] real-valued tensor.

    Takes amplitude from the complex CSI data, flattens, and reshapes
    to match the modality translator's expected input.
    """
    amplitude = np.abs(csi_complex)  # [antennas, subcarriers, samples]
    flat = amplitude.flatten().astype(np.float32)
    n_channels = 64
    spatial = len(flat) // n_channels
    h = w = max(1, int(spatial ** 0.5))
    needed = n_channels * h * w
    if len(flat) < needed:
        flat = np.pad(flat, (0, needed - len(flat)))
    else:
        flat = flat[:needed]
    return torch.from_numpy(flat).view(1, n_channels, h, w)


# --- Tests ---

class TestEndToEndPipeline:
    """Test the full inference pipeline: CSI → modality translation → DensePose → parsed poses."""

    def test_pipeline_produces_valid_output(self, modality_translator, densepose_model, mock_csi_gen):
        """Full pipeline runs without errors and produces expected output structure."""
        csi_raw = mock_csi_gen.generate()
        csi_tensor = csi_to_tensor(csi_raw)

        with torch.no_grad():
            visual_features = modality_translator(csi_tensor)
            poses = densepose_model(visual_features)

        assert isinstance(poses, dict)
        assert 'segmentation' in poses
        assert 'uv_coordinates' in poses
        assert poses['segmentation'].shape[0] == 1  # batch size
        assert poses['segmentation'].shape[1] == 25  # 24 parts + background
        assert poses['uv_coordinates'].shape[1] == 2  # U and V

    def test_pipeline_output_values_in_range(self, modality_translator, densepose_model, mock_csi_gen):
        """UV coordinates should be in [0, 1] (sigmoid applied)."""
        csi_raw = mock_csi_gen.generate()
        csi_tensor = csi_to_tensor(csi_raw)

        with torch.no_grad():
            visual_features = modality_translator(csi_tensor)
            poses = densepose_model(visual_features)

        uv = poses['uv_coordinates']
        assert uv.min() >= 0.0, f"UV min {uv.min()} < 0"
        assert uv.max() <= 1.0, f"UV max {uv.max()} > 1"

    def test_pipeline_segmentation_probabilities(self, modality_translator, densepose_model, mock_csi_gen):
        """Segmentation logits should produce valid probabilities via softmax."""
        csi_raw = mock_csi_gen.generate()
        csi_tensor = csi_to_tensor(csi_raw)

        with torch.no_grad():
            visual_features = modality_translator(csi_tensor)
            poses = densepose_model(visual_features)

        probs = torch.softmax(poses['segmentation'], dim=1)
        # Probabilities sum to 1 along class dimension
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_pipeline_multiple_frames(self, modality_translator, densepose_model, mock_csi_gen):
        """Pipeline handles multiple consecutive frames without error."""
        for frame_idx in range(10):
            csi_raw = mock_csi_gen.generate()
            csi_tensor = csi_to_tensor(csi_raw)

            with torch.no_grad():
                visual_features = modality_translator(csi_tensor)
                poses = densepose_model(visual_features)

            assert poses['segmentation'].shape[0] == 1

    def test_pipeline_deterministic_with_seed(self, modality_translator, densepose_model):
        """Same input produces same output (model in eval mode, no dropout)."""
        torch.manual_seed(42)
        csi_tensor = torch.randn(1, 64, 8, 8)

        with torch.no_grad():
            out1 = densepose_model(modality_translator(csi_tensor))
            out2 = densepose_model(modality_translator(csi_tensor))

        assert torch.allclose(out1['segmentation'], out2['segmentation'])
        assert torch.allclose(out1['uv_coordinates'], out2['uv_coordinates'])


class TestTensorReshapePaths:
    """Test all CSI tensor reshape paths in the pipeline."""

    def test_1d_flat_input(self, modality_translator, densepose_model):
        """1D flat CSI vector is correctly reshaped."""
        flat = torch.randn(64 * 8 * 8)
        n_channels = 64
        spatial = len(flat) // n_channels
        h = w = max(1, int(spatial ** 0.5))
        tensor_4d = flat[:n_channels * h * w].view(1, n_channels, h, w)

        with torch.no_grad():
            visual = modality_translator(tensor_4d)
            poses = densepose_model(visual)
        assert 'segmentation' in poses

    def test_2d_matrix_input(self, modality_translator, densepose_model):
        """2D [antennas, subcarriers] reshaped correctly."""
        mat = torch.randn(3, 56)
        flat = mat.flatten()
        n_channels = 64
        h = w = max(1, int((len(flat) // n_channels) ** 0.5))
        needed = n_channels * h * w
        if len(flat) < needed:
            flat = F.pad(flat, (0, needed - len(flat)))
        tensor_4d = flat[:needed].view(1, n_channels, h, w)

        with torch.no_grad():
            visual = modality_translator(tensor_4d)
            poses = densepose_model(visual)
        assert 'segmentation' in poses

    def test_4d_direct_input(self, modality_translator, densepose_model):
        """4D tensor passed directly."""
        tensor_4d = torch.randn(1, 64, 8, 8)

        with torch.no_grad():
            visual = modality_translator(tensor_4d)
            poses = densepose_model(visual)
        assert poses['uv_coordinates'].shape[1] == 2

    def test_small_spatial_dims(self, modality_translator, densepose_model):
        """Very small spatial dimensions (2x2) still work."""
        tensor_4d = torch.randn(1, 64, 2, 2)

        with torch.no_grad():
            visual = modality_translator(tensor_4d)
            poses = densepose_model(visual)
        assert 'segmentation' in poses

    def test_large_spatial_dims(self, modality_translator, densepose_model):
        """Larger spatial dimensions (32x32) still work."""
        tensor_4d = torch.randn(1, 64, 32, 32)

        with torch.no_grad():
            visual = modality_translator(tensor_4d)
            poses = densepose_model(visual)
        assert 'segmentation' in poses


class TestMockCSIGenerator:
    """Test the mock CSI data generator itself."""

    def test_generate_shape(self, mock_csi_gen):
        """Generated CSI has correct shape."""
        csi = mock_csi_gen.generate()
        assert csi.shape == (3, 64, 100)

    def test_generate_complex(self, mock_csi_gen):
        """Generated CSI is complex-valued."""
        csi = mock_csi_gen.generate()
        assert np.iscomplexobj(csi)

    def test_generate_nonzero(self, mock_csi_gen):
        """Generated CSI has nonzero values."""
        csi = mock_csi_gen.generate()
        assert np.abs(csi).max() > 0

    def test_temporal_variation(self, mock_csi_gen):
        """Consecutive frames differ (temporal evolution)."""
        frame1 = mock_csi_gen.generate()
        frame2 = mock_csi_gen.generate()
        assert not np.allclose(frame1, frame2)

    def test_configure_noise(self, mock_csi_gen):
        """Noise level configuration affects output."""
        mock_csi_gen.configure({'noise_level': 0.0})
        clean = mock_csi_gen.generate()
        mock_csi_gen.configure({'noise_level': 1.0})
        noisy = mock_csi_gen.generate()
        # Noisy signal should have higher variance
        assert np.std(np.abs(noisy)) > np.std(np.abs(clean)) * 0.5
