#!/bin/bash
# Sync models between local and HuggingFace Hub
# Usage: ./scripts/sync-models.sh [push|pull]

HF_REPO="BAS-More/ruview-models"
LOCAL_DIR="models"

case "${1:-pull}" in
  push)
    echo "Pushing local models to HuggingFace..."
    echo "  Uploading csi-ruvllm (your trained models)..."
    cd "$LOCAL_DIR/csi-ruvllm"
    hf upload "$HF_REPO" . csi-ruvllm/ --repo-type model
    echo "  Done. Models available at: https://huggingface.co/$HF_REPO"
    ;;
  pull)
    echo "Pulling models from HuggingFace..."
    echo "  Downloading pre-trained models (ruvnet)..."
    hf download ruv/ruview --local-dir "$LOCAL_DIR/huggingface"
    echo "  Downloading your trained models..."
    hf download "$HF_REPO" --local-dir "$LOCAL_DIR/csi-ruvllm" 2>/dev/null || \
      echo "  (No custom models on HF yet — run 'sync-models.sh push' first)"
    echo "  Done."
    ;;
  *)
    echo "Usage: $0 [push|pull]"
    echo "  push  Upload local models to HuggingFace"
    echo "  pull  Download models from HuggingFace to local"
    ;;
esac
