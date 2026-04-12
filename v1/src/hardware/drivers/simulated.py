"""
Simulated Phase A sensor backends for development and testing.

Each simulator generates realistic synthetic data matching the real
sensor's output format, allowing the full fusion pipeline to run
end-to-end without hardware.

Usage::

    from v1.src.hardware.drivers.simulated import SimulatedSensorSuite
    suite = SimulatedSensorSuite()
    registry = suite.create_registry()
    await registry.auto_detect()  # all 6 simulated sensors connect
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional, Set

import numpy as np

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)


# ---------------------------------------------------------------------------
# Simulated LD2450 — 24 GHz radar with moving targets
# ---------------------------------------------------------------------------

class SimulatedLD2450(SensorDriver):
    """Simulated HLK-LD2450 with 1-3 targets moving in a room."""

    def __init__(self, sensor_id: str = "ld2450", **kw: Any) -> None:
        super().__init__(sensor_id, **kw)
        self._tick = 0
        self._targets: List[Dict[str, float]] = []
        self._rng = random.Random(42)
        # Spawn 1-2 initial targets
        for _ in range(self._rng.randint(1, 2)):
            self._targets.append(self._new_target())

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {SensorCapability.PRESENCE, SensorCapability.DISTANCE, SensorCapability.MULTI_TARGET}

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.UART

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        self._tick += 1
        self._move_targets()
        targets = []
        for t in self._targets:
            dist = int(math.sqrt(t["x"] ** 2 + t["y"] ** 2))
            targets.append({
                "x_mm": int(t["x"]),
                "y_mm": int(t["y"]),
                "speed_mm_s": int(t["vx"]),
                "distance_mm": dist,
            })
        nearest = min((t["distance_mm"] for t in targets), default=0)
        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "person_present": len(targets) > 0,
                "target_count": len(targets),
                "targets": targets,
                "nearest_distance_mm": nearest,
            },
        )

    def _new_target(self) -> Dict[str, float]:
        return {
            "x": self._rng.uniform(-2000, 2000),
            "y": self._rng.uniform(500, 3000),
            "vx": self._rng.uniform(-200, 200),
            "vy": self._rng.uniform(-100, 100),
        }

    def _move_targets(self) -> None:
        for t in self._targets:
            t["x"] += t["vx"] * 0.1
            t["y"] += t["vy"] * 0.1
            # Bounce off walls
            if abs(t["x"]) > 3000:
                t["vx"] *= -1
            if t["y"] < 200 or t["y"] > 4000:
                t["vy"] *= -1
            # Random velocity jitter
            t["vx"] += self._rng.gauss(0, 20)
            t["vy"] += self._rng.gauss(0, 10)
        # Occasionally add/remove targets
        if self._tick % 100 == 0 and self._rng.random() < 0.3:
            if len(self._targets) < 3:
                self._targets.append(self._new_target())
            elif len(self._targets) > 1 and self._rng.random() < 0.5:
                self._targets.pop()


# ---------------------------------------------------------------------------
# Simulated ENS160 — air quality with slow drift
# ---------------------------------------------------------------------------

class SimulatedENS160(SensorDriver):
    """Simulated ENS160 with slowly drifting TVOC/eCO2."""

    def __init__(self, sensor_id: str = "ens160", **kw: Any) -> None:
        super().__init__(sensor_id, **kw)
        self._tvoc = 120.0
        self._eco2 = 450.0
        self._rng = random.Random(43)

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {SensorCapability.AIR_QUALITY}

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        # Brownian drift
        self._tvoc = max(0, self._tvoc + self._rng.gauss(0, 5))
        self._eco2 = max(400, self._eco2 + self._rng.gauss(0, 10))
        aqi = 1 if self._tvoc < 200 else (2 if self._tvoc < 500 else 3)
        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "tvoc_ppb": int(self._tvoc),
                "eco2_ppm": int(self._eco2),
                "aqi": aqi,
                "validity_flag": 3,
            },
        )


# ---------------------------------------------------------------------------
# Simulated AMG8833 — 8x8 thermal with body heat spot
# ---------------------------------------------------------------------------

class SimulatedAMG8833(SensorDriver):
    """Simulated AMG8833 with a body-shaped hot spot on the 8x8 grid."""

    def __init__(self, sensor_id: str = "amg8833", **kw: Any) -> None:
        super().__init__(sensor_id, **kw)
        self._ambient = 22.0
        self._body_x = 4.0
        self._body_y = 4.0
        self._rng = random.Random(44)

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {SensorCapability.THERMAL_IMAGE, SensorCapability.TEMPERATURE, SensorCapability.PRESENCE}

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        # Body slowly drifts
        self._body_x += self._rng.gauss(0, 0.2)
        self._body_y += self._rng.gauss(0, 0.2)
        self._body_x = max(1, min(6, self._body_x))
        self._body_y = max(1, min(6, self._body_y))

        grid = []
        max_t = self._ambient
        for row in range(8):
            pixels = []
            for col in range(8):
                dx = col - self._body_x
                dy = row - self._body_y
                dist = math.sqrt(dx * dx + dy * dy)
                # Gaussian body heat blob
                body_heat = 14.0 * math.exp(-dist * dist / 3.0)
                t = self._ambient + body_heat + self._rng.gauss(0, 0.3)
                pixels.append(round(t, 2))
                max_t = max(max_t, t)
            grid.append(pixels)

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "grid": grid,
                "thermistor_c": round(self._ambient + self._rng.gauss(0, 0.1), 2),
                "max_c": round(max_t, 2),
                "min_c": round(self._ambient - 0.5, 2),
                "avg_c": round(self._ambient + 2.0, 2),
                "person_present": max_t > self._ambient + 3.0,
            },
        )


# ---------------------------------------------------------------------------
# Simulated BME688 — environment with diurnal cycle
# ---------------------------------------------------------------------------

class SimulatedBME688(SensorDriver):
    """Simulated BME688 with slowly varying environment."""

    def __init__(self, sensor_id: str = "bme688", **kw: Any) -> None:
        super().__init__(sensor_id, **kw)
        self._t0 = time.monotonic()
        self._rng = random.Random(45)

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.TEMPERATURE, SensorCapability.HUMIDITY,
            SensorCapability.PRESSURE, SensorCapability.GAS_RESISTANCE,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2C

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        elapsed = time.monotonic() - self._t0
        # Simulate slow diurnal-like variation (accelerated: 1 cycle per 60s)
        phase = elapsed / 60.0 * 2 * math.pi
        temp = 22.0 + 3.0 * math.sin(phase) + self._rng.gauss(0, 0.2)
        hum = 45.0 + 10.0 * math.cos(phase) + self._rng.gauss(0, 1)
        press = 1013.25 + 2.0 * math.sin(phase * 0.3) + self._rng.gauss(0, 0.1)
        gas = 50000 + 20000 * math.sin(phase * 0.5) + self._rng.gauss(0, 1000)

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "temperature_c": round(temp, 2),
                "humidity_pct": round(max(0, min(100, hum)), 1),
                "pressure_hpa": round(press, 2),
                "gas_resistance_ohm": round(max(1000, gas), 0),
                "gas_valid": True,
                "heater_stable": True,
            },
        )


# ---------------------------------------------------------------------------
# Simulated MR60BHA2 — 60 GHz mmWave with vital signs
# ---------------------------------------------------------------------------

class SimulatedMR60BHA2(SensorDriver):
    """Simulated MR60BHA2 with sine-wave heart rate and breathing."""

    def __init__(self, sensor_id: str = "mr60bha2", **kw: Any) -> None:
        super().__init__(sensor_id, **kw)
        self._t0 = time.monotonic()
        self._rng = random.Random(46)
        self._base_hr = 72.0
        self._base_br = 16.0
        self._present = True

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.HEART_RATE, SensorCapability.BREATHING_RATE,
            SensorCapability.PRESENCE, SensorCapability.DISTANCE,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.UART

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        elapsed = time.monotonic() - self._t0
        # Heart rate with slow variation + noise
        hr = self._base_hr + 5 * math.sin(elapsed / 30.0) + self._rng.gauss(0, 1.0)
        br = self._base_br + 2 * math.sin(elapsed / 45.0) + self._rng.gauss(0, 0.3)
        dist = 80.0 + 30 * math.sin(elapsed / 20.0)  # person sways

        # Occasionally toggle presence
        if self._rng.random() < 0.002:
            self._present = not self._present

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "heart_rate_bpm": round(hr, 1) if self._present else 0.0,
                "breathing_rate_bpm": round(br, 1) if self._present else 0.0,
                "distance_cm": round(dist, 1) if self._present else 0.0,
                "person_present": self._present,
            },
        )


# ---------------------------------------------------------------------------
# Simulated INMP441 — MEMS microphone with ambient noise
# ---------------------------------------------------------------------------

class SimulatedINMP441(SensorDriver):
    """Simulated INMP441 with ambient noise fluctuations."""

    def __init__(self, sensor_id: str = "inmp441", **kw: Any) -> None:
        super().__init__(sensor_id, **kw)
        self._rng = random.Random(47)
        self._base_spl = 40.0  # quiet room

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {SensorCapability.AUDIO_LEVEL, SensorCapability.AUDIO_STREAM}

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2S

    async def _connect(self) -> None:
        pass

    async def _disconnect(self) -> None:
        pass

    async def _read(self) -> SensorReading:
        # Ambient noise with occasional spikes (door slam, speech)
        spl = self._base_spl + self._rng.gauss(0, 2)
        if self._rng.random() < 0.05:
            spl += self._rng.uniform(10, 30)  # transient event
        spl = max(20, min(100, spl))

        # Generate a small synthetic PCM block (256 samples at 16kHz)
        samples = np.random.default_rng(int(time.monotonic() * 1000) % 2**31).normal(
            0, spl * 100, size=256
        ).astype(np.int32)

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "db_spl": round(spl, 1),
                "rms": float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))),
                "peak": float(np.max(np.abs(samples))),
                "sample_rate": 16000,
                "block_size": 256,
            },
        )


# ---------------------------------------------------------------------------
# Suite factory
# ---------------------------------------------------------------------------

class SimulatedSensorSuite:
    """Factory that creates a full set of simulated Phase A sensors.

    Usage::

        suite = SimulatedSensorSuite()
        registry = await suite.create_registry()
        # All 6 sensors are connected and producing synthetic data
        readings = await registry.read_all()
    """

    DRIVERS = [
        SimulatedLD2450,
        SimulatedENS160,
        SimulatedAMG8833,
        SimulatedBME688,
        SimulatedMR60BHA2,
        SimulatedINMP441,
    ]

    async def create_registry(self):
        """Create and populate a SensorRegistry with all simulated sensors."""
        from v1.src.hardware.sensor_registry import SensorRegistry

        reg = SensorRegistry()
        for cls in self.DRIVERS:
            drv = cls()
            await reg.register(drv)
        return reg
