# RuView Home Testing Setup Guide

## Hardware Shopping List

| Item | Qty | Specs | Est. Cost | Where to Buy |
|------|-----|-------|-----------|-------------|
| ESP32-S3-DevKitC-1 | 3 | N16R8 (16MB flash, 8MB PSRAM) | $12 each | Amazon, AliExpress, Mouser |
| 2.4GHz IPEX antenna | 6 | 2dBi, IPEX/U.FL connector | $2 each | Amazon |
| USB-C data cable | 3 | 1m, data-capable (not charge-only) | $4 each | Amazon |
| USB webcam | 1 | 1080p, Logitech C920 or similar | $30-40 | Amazon |
| Small tripod | 1 | For webcam mounting | $12 | Amazon |
| **Total** | | | **~$107** | |

> **Important**: Must be ESP32-**S3** variant. Original ESP32 and ESP32-C3 are single-core
> and cannot run the CSI DSP pipeline.

## Room Setup

### Sensor Placement

```
                    Router (existing)
                         |
    ┌────────────────────┼────────────────────┐
    │                    |                    │
    │  ESP32-A ·····················  ESP32-B │
    │  (corner)                    (corner)   │
    │                                         │
    │              SENSING AREA               │
    │         (where person moves)            │
    │                                         │
    │                                         │
    │                · ESP32-C                │
    │              (mid-wall)                 │
    │                                         │
    │  [Webcam on tripod]                     │
    │  (aimed at sensing area)                │
    └─────────────────────────────────────────┘

    Recommended room: 3m x 4m to 5m x 6m
    Sensor spacing: 2-4 meters apart
    Webcam: positioned to see entire sensing area
```

### Network Requirements
- All ESP32s and your PC must be on the **same WiFi network**
- 2.4GHz band (ESP32-S3 does not support 5GHz for CSI)
- Stable router (avoid mesh systems that roam between nodes)

## Software Setup

### 1. Install ESP-IDF (one time)

Follow Espressif's guide: https://docs.espressif.com/projects/esp-idf/en/v5.4/esp32s3/get-started/

```bash
# Windows: Use ESP-IDF installer from https://dl.espressif.com/dl/esp-idf/
# Linux/Mac:
mkdir -p ~/esp
cd ~/esp
git clone -b v5.4 --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
./install.sh esp32s3
source export.sh
```

### 2. Flash Firmware

```bash
cd firmware/esp32-csi-node

# Build for 8MB flash
cp sdkconfig.defaults.template sdkconfig.defaults
idf.py set-target esp32s3
idf.py build

# Flash each ESP32 (change COM port for each)
idf.py -p COM7 flash    # ESP32-A
idf.py -p COM8 flash    # ESP32-B
idf.py -p COM9 flash    # ESP32-C
```

### 3. Provision WiFi

```bash
# For each ESP32:
python firmware/esp32-csi-node/provision.py \
  --port COM7 \
  --ssid "YourWiFiSSID" \
  --password "YourWiFiPassword" \
  --target-ip $(hostname -I | awk '{print $1}')  # your PC's IP
```

### 4. Verify CSI Data Streaming

```bash
# Monitor serial output from one ESP32
python -m serial.tools.miniterm COM7 115200

# You should see lines like:
# CSI: node_id=esp32-a, rssi=-45, subcarriers=64, ...
```

### 5. Run Calibration

```bash
# Empty room calibration (30 seconds, nobody in the room)
python scripts/calibrate_room.py --duration 30

# Walking path calibration (walk around the room)
python scripts/calibrate_room.py --mode walking --duration 60
```

### 6. Collect Training Data

```bash
# Simultaneous CSI + camera recording
python scripts/collect_training_data.py \
  --duration 30m \
  --output data/recordings/session1/ \
  --camera 0  # webcam index, usually 0
```

During collection:
- Walk around the sensing area in varied paths
- Stand, sit, wave, turn, lie down
- Cover different positions in the room
- The script shows real-time frame counts for both CSI and camera

### 7. Train Model

```bash
cd v1

# Preprocess collected data
python -m src.training.preprocess \
  --input ../data/recordings/session1/ \
  --output ../data/processed/ \
  --keypoints ../data/recordings/session1/keypoints.jsonl

# Train (50 epochs, ~30 min on CPU)
python -m src.training.train_pose \
  --data ../data/processed/ \
  --output ../models/ \
  --epochs 50

# Evaluate
python -m src.training.evaluate \
  --model ../models/best.pt \
  --data ../data/processed/test/
```

### 8. Run Live System

```bash
# Start the server with trained model
cd v1
SECRET_KEY=dev-secret python -m uvicorn src.api.main:app \
  --host 0.0.0.0 --port 8000

# Open browser to http://localhost:8000/docs for API
# Open http://localhost:8000/ for web UI (if available)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ESP32 not detected on USB | Use a **data** USB-C cable, not charge-only |
| No CSI data streaming | Check WiFi provisioning, verify same network |
| Low CSI frame rate | Reduce distance between sensors, check antenna connections |
| Webcam not found | Try `--camera 1` or install `opencv-python` |
| Training loss not decreasing | Collect more data (minimum 5000 paired frames) |
| Poor accuracy after training | Recollect data at different times of day, add more activities |
