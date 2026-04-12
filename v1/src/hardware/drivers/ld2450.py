"""
HLK-LD2450 24 GHz radar — presence, distance, multi-target tracking.

Wraps the ``hlk-ld2450`` library (csRon/HLK-LD2450) which speaks the
HLK binary UART protocol at 256000 baud.

Install:  pip install hlk-ld2450
Hardware: HLK-LD2450 module on UART (TX/RX, 3.3 V logic).
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)


@dataclass
class LD2450Target:
    """Single tracked target from the LD2450."""
    x_mm: int
    y_mm: int
    speed_mm_s: int
    resolution_mm: int


# LD2450 binary frame constants
_FRAME_HEADER = bytes([0xAA, 0xFF, 0x03, 0x00])
_FRAME_FOOTER = bytes([0x55, 0xCC])
_TARGET_SIZE = 8  # 8 bytes per target slot
_MAX_TARGETS = 3


class LD2450Driver(SensorDriver):
    """Driver for HLK-LD2450 24 GHz multi-target radar.

    Parameters
    ----------
    sensor_id : str
        Unique identifier for this sensor instance.
    port : str
        Serial port path (e.g. ``/dev/ttyUSB0`` or ``COM3``).
    baudrate : int
        UART baud rate. Default 256000 per HLK spec.
    """

    def __init__(
        self,
        sensor_id: str = "ld2450",
        port: str = "/dev/ttyUSB0",
        baudrate: int = 256_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(sensor_id, **kwargs)
        self._port = port
        self._baudrate = baudrate
        self._serial: Any = None  # serial.Serial instance
        self._reader_task: Optional[asyncio.Task] = None
        self._targets: List[LD2450Target] = []
        self._person_present = False

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.PRESENCE,
            SensorCapability.DISTANCE,
            SensorCapability.MULTI_TARGET,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.UART

    async def _connect(self) -> None:
        try:
            import serial
        except ImportError:
            raise ImportError(
                "hlk-ld2450 requires pyserial. Install: pip install hlk-ld2450"
            )

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=0.1,
        )
        if not self._serial.is_open:
            self._serial.open()

        # Flush stale data
        self._serial.reset_input_buffer()
        self._logger.info("LD2450 opened on %s @ %d baud", self._port, self._baudrate)

    async def _disconnect(self) -> None:
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    async def _read(self) -> SensorReading:
        """Read one complete target frame from the LD2450.

        The LD2450 sends frames at ~10 Hz.  Each frame contains up to 3
        target slots.  An empty slot has x=y=speed=resolution=0.
        """
        raw = await asyncio.get_event_loop().run_in_executor(
            None, self._read_frame_sync
        )
        targets = self._parse_frame(raw)
        self._targets = targets
        self._person_present = len(targets) > 0

        values: Dict[str, Any] = {
            "person_present": self._person_present,
            "target_count": len(targets),
            "targets": [
                {
                    "x_mm": t.x_mm,
                    "y_mm": t.y_mm,
                    "speed_mm_s": t.speed_mm_s,
                    "distance_mm": int((t.x_mm**2 + t.y_mm**2) ** 0.5),
                }
                for t in targets
            ],
        }

        if targets:
            nearest = min(targets, key=lambda t: t.x_mm**2 + t.y_mm**2)
            values["nearest_distance_mm"] = int(
                (nearest.x_mm**2 + nearest.y_mm**2) ** 0.5
            )

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values=values,
            raw=raw,
        )

    # -- internal -------------------------------------------------------------

    def _read_frame_sync(self) -> bytes:
        """Block until a full LD2450 frame is received."""
        buf = bytearray()
        while True:
            b = self._serial.read(1)
            if not b:
                continue
            buf.append(b[0])
            # Look for frame header
            if len(buf) >= 4 and buf[-4:] == bytearray(_FRAME_HEADER):
                # Read the target data (3 targets * 8 bytes) + 2 footer bytes
                payload = self._serial.read(_MAX_TARGETS * _TARGET_SIZE + 2)
                return bytes(buf[-4:] + payload)

    @staticmethod
    def _parse_frame(raw: bytes) -> List[LD2450Target]:
        """Parse a raw LD2450 frame into target list."""
        targets: List[LD2450Target] = []
        offset = 4  # skip header

        for _ in range(_MAX_TARGETS):
            if offset + _TARGET_SIZE > len(raw):
                break
            x, y, speed, res = struct.unpack_from("<hhhH", raw, offset)
            offset += _TARGET_SIZE
            # Empty slot: all zeros
            if x == 0 and y == 0 and speed == 0 and res == 0:
                continue
            targets.append(LD2450Target(x_mm=x, y_mm=y, speed_mm_s=speed, resolution_mm=res))

        return targets
