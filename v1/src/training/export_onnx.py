"""
Export trained WiFi-DensePose models to ONNX format.

Usage:
    python -m src.training.export_onnx --model models/best.pt --output models/ruview.onnx
"""

import argparse
import logging
import os

import torch

logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 17,
    device_name: str = 'cpu',
):
    """Export trained models to a single combined ONNX file.

    Args:
        model_path: Path to PyTorch checkpoint (.pt).
        output_path: Output ONNX file path.
        opset_version: ONNX opset version.
        device_name: Device to use for tracing.
    """
    device = torch.device(device_name)

    # Build models and load weights
    from src.training.train_pose import build_models
    mt, dp = build_models(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    mt.load_state_dict(checkpoint['mt_state_dict'])
    dp.load_state_dict(checkpoint['dp_state_dict'])
    mt.eval()
    dp.eval()
    logger.info(f"Models loaded from {model_path}")

    # Create a combined wrapper for export
    class CombinedModel(torch.nn.Module):
        def __init__(self, modality_translator, densepose_head):
            super().__init__()
            self.mt = modality_translator
            self.dp = densepose_head

        def forward(self, csi_input):
            visual = self.mt(csi_input)
            output = self.dp(visual)
            return output['segmentation'], output['uv_coordinates']

    combined = CombinedModel(mt, dp).to(device)
    combined.eval()

    # Dummy input for tracing
    dummy_input = torch.randn(1, 64, 8, 8, device=device)

    # Workaround: PyTorch ONNX exporter prints emoji checkmarks that crash
    # on Windows cp1252 consoles. Force UTF-8 for stdout/stderr.
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Export
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch.onnx.export(
        combined,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=['csi_input'],
        output_names=['segmentation', 'uv_coordinates'],
        dynamic_axes={
            'csi_input': {0: 'batch_size'},
            'segmentation': {0: 'batch_size'},
            'uv_coordinates': {0: 'batch_size'},
        },
    )

    file_size = os.path.getsize(output_path) / 1024
    logger.info(f"ONNX model exported to {output_path} ({file_size:.1f} KB)")

    # Validate
    try:
        import onnx
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        logger.info("ONNX model validation: PASS")
    except ImportError:
        logger.warning("onnx package not installed, skipping validation")
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model', required=True, help='PyTorch checkpoint path')
    parser.add_argument('--output', default='models/ruview.onnx', help='Output ONNX path')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    export_to_onnx(args.model, args.output, args.opset, args.device)


if __name__ == '__main__':
    main()
