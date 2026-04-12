"""
Sensor data recorder and playback for offline analysis.

Records ``FusedSensingResult`` frames to JSONL files and replays them
through the fusion pipeline or directly to the WebSocket server.

Usage::

    # Record
    recorder = SensorRecorder("data/recordings/session-001.jsonl")
    recorder.start()
    recorder.record_frame(fused_result)  # called each tick
    recorder.stop()

    # Playback
    player = SensorPlayer("data/recordings/session-001.jsonl")
    async for frame in player.play(speed=1.0):
        # frame is a dict matching the fusion JSON schema
        ws_server._last_fused = frame
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

logger = logging.getLogger(__name__)


class SensorRecorder:
    """Records fused sensor data to a JSONL file.

    Each line is a JSON object with ``timestamp``, ``fusion``, and
    ``wifi`` keys.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._file = None
        self._frame_count = 0
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Open the recording file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w", encoding="utf-8")
        self._start_time = time.time()
        self._frame_count = 0
        logger.info("Recording started: %s", self._path)

    def stop(self) -> None:
        """Close the recording file."""
        if self._file:
            self._file.close()
            self._file = None
        elapsed = time.time() - (self._start_time or time.time())
        logger.info(
            "Recording stopped: %d frames in %.1fs -> %s",
            self._frame_count, elapsed, self._path,
        )

    def record_frame(self, fused_result) -> None:
        """Record one FusedSensingResult frame.

        Parameters
        ----------
        fused_result : FusedSensingResult
            The fused sensing result from MultiSensorBackend.fuse().
        """
        if not self._file:
            return

        frame = {
            "timestamp": time.time(),
            "frame_id": self._frame_count,
            "fusion": {
                "presence": fused_result.presence,
                "presence_sources": fused_result.presence_sources,
                "fused_confidence": fused_result.fused_confidence,
                "heart_rate_bpm": fused_result.heart_rate_bpm,
                "breathing_rate_bpm": fused_result.breathing_rate_bpm,
                "nearest_distance_mm": fused_result.nearest_distance_mm,
                "target_count": fused_result.target_count,
                "temperature_c": fused_result.temperature_c,
                "humidity_pct": fused_result.humidity_pct,
                "pressure_hpa": fused_result.pressure_hpa,
                "tvoc_ppb": fused_result.tvoc_ppb,
                "eco2_ppm": fused_result.eco2_ppm,
                "aqi": fused_result.aqi,
                "thermal_max_c": fused_result.thermal_max_c,
                "thermal_presence": fused_result.thermal_presence,
                "db_spl": fused_result.db_spl,
            },
            "wifi": {
                "motion_level": fused_result.wifi.motion_level.value,
                "confidence": fused_result.wifi.confidence,
                "presence": fused_result.wifi.presence_detected,
            },
            "sensor_readings": fused_result.sensor_readings,
        }

        self._file.write(json.dumps(frame, default=str) + "\n")
        self._frame_count += 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._file is not None


class SensorPlayer:
    """Replays recorded JSONL sensor data.

    Parameters
    ----------
    path : str or Path
        Path to the JSONL recording file.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._frames: list[Dict[str, Any]] = []

    def load(self) -> int:
        """Load all frames from the JSONL file.

        Returns
        -------
        int
            Number of frames loaded.
        """
        self._frames = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._frames.append(json.loads(line))
        logger.info("Loaded %d frames from %s", len(self._frames), self._path)
        return len(self._frames)

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def get_frame(self, index: int) -> Dict[str, Any]:
        """Get a specific frame by index."""
        return self._frames[index]

    async def play(self, speed: float = 1.0) -> AsyncIterator[Dict[str, Any]]:
        """Replay frames at the given speed multiplier.

        Parameters
        ----------
        speed : float
            Playback speed (1.0 = real-time, 2.0 = 2x speed, 0 = no delay).

        Yields
        ------
        dict
            Each frame's fusion data.
        """
        if not self._frames:
            self.load()

        prev_ts = None
        for frame in self._frames:
            ts = frame.get("timestamp", 0)
            if prev_ts is not None and speed > 0:
                delay = (ts - prev_ts) / speed
                if delay > 0:
                    await asyncio.sleep(delay)
            prev_ts = ts
            yield frame

    def as_fused_result(self, frame: Dict[str, Any]):
        """Convert a recorded frame back to a FusedSensingResult.

        Parameters
        ----------
        frame : dict
            A frame dict from the JSONL recording.

        Returns
        -------
        FusedSensingResult
        """
        from v1.src.sensing.multi_sensor_backend import FusedSensingResult
        from v1.src.sensing.classifier import MotionLevel, SensingResult

        wifi_data = frame.get("wifi", {})
        wifi_result = SensingResult(
            motion_level=MotionLevel(wifi_data.get("motion_level", "absent")),
            confidence=wifi_data.get("confidence", 0),
            presence_detected=wifi_data.get("presence", False),
            rssi_variance=0,
            motion_band_energy=0,
            breathing_band_energy=0,
            n_change_points=0,
        )

        f = frame.get("fusion", {})
        return FusedSensingResult(
            wifi=wifi_result,
            presence=f.get("presence", False),
            presence_sources=f.get("presence_sources", []),
            fused_confidence=f.get("fused_confidence", 0),
            heart_rate_bpm=f.get("heart_rate_bpm"),
            breathing_rate_bpm=f.get("breathing_rate_bpm"),
            nearest_distance_mm=f.get("nearest_distance_mm"),
            target_count=f.get("target_count", 0),
            temperature_c=f.get("temperature_c"),
            humidity_pct=f.get("humidity_pct"),
            pressure_hpa=f.get("pressure_hpa"),
            tvoc_ppb=f.get("tvoc_ppb"),
            eco2_ppm=f.get("eco2_ppm"),
            aqi=f.get("aqi"),
            thermal_max_c=f.get("thermal_max_c"),
            thermal_presence=f.get("thermal_presence"),
            db_spl=f.get("db_spl"),
            sensor_readings=frame.get("sensor_readings", {}),
        )
