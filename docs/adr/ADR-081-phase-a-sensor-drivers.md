# ADR-081: Phase A Multi-Modal Sensor Drivers

**Status:** Accepted
**Date:** 2026-04-12
**Deciders:** @BAS-More
**Related:** ADR-063 (mmWave sensor fusion), ADR-021 (vital sign extraction), ADR-013 (commodity sensing)

## Context

RuView's sensing pipeline relied solely on WiFi CSI/RSSI for presence, motion, and pose estimation. ADR-063 proposed adding 60 GHz mmWave radar. This ADR extends the approach to a full multi-modal sensor suite using commodity hardware and existing open-source libraries.

## Decision

Implement Phase A sensor drivers as thin Python wrappers around proven libraries, with a uniform `SensorDriver` ABC, auto-detection registry, and a `MultiSensorBackend` fusion engine.

### Sensors

| Sensor | Bus | Library | Capabilities | Cost |
|--------|-----|---------|-------------|------|
| HLK-LD2450 | UART 256k | csRon/HLK-LD2450 (pyserial) | Presence, distance, 3-target tracking | ~$3 |
| ENS160 | I2C 0x53 | `ens160` pip | TVOC, eCO2, AQI | ~$5 |
| AMG8833 | I2C 0x69 | `adafruit-circuitpython-amg88xx` | 8x8 thermal grid, presence | ~$15 |
| BME688 | I2C 0x76 | `bme68x` (pi3g) | Temp, humidity, pressure, gas | ~$10 |
| MR60BHA2 | UART 115200 | Ported from Seeed-mmWave-library | Heart rate, breathing, presence, distance | ~$15 |
| INMP441 | I2S | MicroPython `machine.I2S` / `sounddevice` | SPL (dB), raw PCM stream | ~$2 |

### Architecture

```
Hardware -> SensorDriver ABC -> SensorRegistry -> MultiSensorBackend -> WebSocket Server -> UI
                                  auto_detect()      fuse()              JSON broadcast     fusion panel
```

#### Key Components

1. **`v1/src/hardware/base.py`** - `SensorDriver` ABC with `connect/disconnect/read` lifecycle, `SensorCapability` enum, `SensorReading` dataclass
2. **`v1/src/hardware/drivers/`** - 6 driver modules, each ~150 lines
3. **`v1/src/hardware/sensor_registry.py`** - Probes all sensors, registers responders, `read_all()` concurrent
4. **`v1/src/sensing/multi_sensor_backend.py`** - Fuses WiFi `SensingResult` + Phase A readings into `FusedSensingResult`
5. **`v1/src/sensing/ws_server.py`** - Broadcasts fusion data as structured JSON

#### Fusion Rules (per ADR-063)

- **Presence**: confirmed if ANY modality detects (WiFi OR mmWave OR radar OR thermal)
- **Confidence**: boosted by multi-modal agreement (1 source: 1x, 2: 1.15x, 3+: 1.3x)
- **Vitals**: prefer MR60BHA2 (mmWave) when available
- **Distance**: prefer LD2450 (multi-target), fall back to MR60BHA2
- **Environment**: BME688 + ENS160 combined

#### WebSocket Message Structure

```json
{
  "type": "sensing_update",
  "classification": { "presence": true, "confidence": 0.75 },
  "fusion": {
    "presence": true,
    "presence_sources": ["wifi", "mmwave_60ghz", "radar_24ghz"],
    "fused_confidence": 0.92,
    "heart_rate_bpm": 72.0,
    "breathing_rate_bpm": 16.5,
    "nearest_distance_mm": 1200,
    "target_count": 1,
    "environment": { "temperature_c": 23.0, "humidity_pct": 48.0, "pressure_hpa": 1013.0 },
    "air_quality": { "tvoc_ppb": 120, "eco2_ppm": 500, "aqi": 2 },
    "thermal": { "max_c": 35.8, "presence": true },
    "audio": { "db_spl": 52.0 }
  }
}
```

## Consequences

- **Positive**: 6 new sensing modalities from ~$50 total hardware. Non-blocking auto-detect means the system degrades gracefully to WiFi-only.
- **Positive**: 50 new tests (27 driver + 19 fusion + 4 WS integration). Full suite: 570 passed.
- **Negative**: Adds `pyserial`, `ens160`, `adafruit-circuitpython-amg88xx`, `bme68x` as optional dependencies.
- **Risk**: I2C sensors require Linux/RPi with Blinka; INMP441 needs MicroPython or ALSA. Development on Windows uses mock/simulated fallbacks.
