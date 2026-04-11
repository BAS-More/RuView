#!/bin/bash
# RuView Hetzner Deployment Script
# Run this on the server: bash deploy-hetzner.sh
set -e

echo "=== RuView Model Deployment ==="
echo ""

# 1. Install HuggingFace CLI
echo "[1/5] Installing HuggingFace CLI..."
pip install -q huggingface_hub 2>/dev/null || pip3 install -q huggingface_hub 2>/dev/null
echo "  Done."

# 2. Create directories
echo "[2/5] Creating model directories..."
mkdir -p /opt/models/ruview/{pretrained,wiflow,csi-ruvllm}
echo "  Done."

# 3. Download pre-trained models (public)
echo "[3/5] Downloading pre-trained models from HuggingFace..."
hf download ruv/ruview --local-dir /opt/models/ruview/pretrained --quiet
echo "  Done."

# 4. Create sync script
echo "[4/5] Creating auto-sync script..."
cat > /opt/ecosystem/sync-ruview-models.sh << 'SYNCEOF'
#!/bin/bash
set -euo pipefail
echo "[$(date)] Syncing RuView models from HuggingFace..."
hf download ruv/ruview --local-dir /opt/models/ruview/pretrained --quiet
hf download aviben770/ruview-models --local-dir /opt/models/ruview/csi-ruvllm --quiet 2>/dev/null || true
echo "[$(date)] Sync complete. $(du -sh /opt/models/ruview/ | cut -f1) total"
SYNCEOF
chmod +x /opt/ecosystem/sync-ruview-models.sh

# 5. Add cron
echo "[5/5] Setting up auto-sync cron (every 6 hours)..."
(crontab -l 2>/dev/null | grep -v ruview; echo "0 */6 * * * /opt/ecosystem/sync-ruview-models.sh >> /var/log/ruview-model-sync.log 2>&1") | crontab -
echo "  Done."

# Verify
echo ""
echo "=== Verification ==="
echo "Pre-trained models:"
ls /opt/models/ruview/pretrained/*.safetensors /opt/models/ruview/pretrained/*.json 2>/dev/null | while read f; do echo "  $(basename $f) ($(stat -c%s $f) bytes)"; done
echo ""
echo "Cron:"
crontab -l | grep ruview
echo ""
echo "Total size: $(du -sh /opt/models/ruview/ | cut -f1)"
echo ""
echo "=== Deployment Complete ==="
echo ""
echo "To also sync your private models, run:"
echo "  hf auth login    (paste your HF token)"
echo "  hf download aviben770/ruview-models --local-dir /opt/models/ruview/csi-ruvllm"
