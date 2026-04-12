"""
Seeed MR60BHA2 — 60 GHz FMCW mmWave radar (heart rate, breathing, presence).

Ported from the Seeed-mmWave-library Arduino/C++ implementation.
Speaks the Seeed mmWave binary UART protocol at 115200 baud.

The frame format matches the C firmware driver in
``firmware/esp32-csi-node/main/mmwave_sensor.c``:

    [0]    SOF = 0x01
    [1-2]  Frame ID (uint16, big-endian)
    [3-4]  Data Length (uint16, big-endian)
    [5-6]  Frame Type (uint16, big-endian)
    [7]    Header Checksum = ~XOR(bytes 0..6) & 0xFF
    [8..N] Payload (N = data_length)
    [N+1]  Data Checksum = ~XOR(payload bytes) & 0xFF

Frame types:
    0x0A14 = breathing rate
    0x0A15 = heart rate
    0x0A16 = distance
    0x0F09 = presence

Install:  pip install pyserial
Hardware: Seeed MR60BHA2 on UART (TX/RX, 3.3 V logic, 115200 baud).
"""

from __future__ import annotations

import asyncio
import struct
from typing import Any, Dict, Optional, Set

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)

# Protocol constants (ported from Seeed-mmWave-library)
_SOF = 0x01
_TYPE_BREATHING = 0x0A14
_TYPE_HEART_RATE = 0x0A15
_TYPE_DISTANCE = 0x0A16
_TYPE_PRESENCE = 0x0F09
_MAX_PAYLOAD = 30  # Sanity limit from Seeed Arduino lib


class MR60BHA2Driver(SensorDriver):
    """Driver for Seeed MR60BHA2 60 GHz mmWave vital-sign radar.

    Parameters
    ----------
    sensor_id : str
        Unique identifier.
    port : str
        Serial port path.
    baudrate : int
        UART baud rate. Default 115200 per MR60BHA2 spec.
    """

    def __init__(
        self,
        sensor_id: str = "mr60bha2",
        port: str = "/dev/ttyUSB1",
        baudrate: int = 115_200,
        **kwargs: Any,
    ) -> None:
        super().__init__(sensor_id, **kwargs)
        self._port = port
        self._baudrate = baudrate
        self._serial: Any = None

        # Accumulated state (updated incrementally per frame type)
        self._heart_rate_bpm: float = 0.0
        self._breathing_rate: float = 0.0
        self._distance_cm: float = 0.0
        self._person_present: bool = False

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.HEART_RATE,
            SensorCapability.BREATHING_RATE,
            SensorCapability.PRESENCE,
            SensorCapability.DISTANCE,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.UART

    async def _connect(self) -> None:
        try:
            import serial
        except ImportError:
            raise ImportError("MR60BHA2 driver requires: pip install pyserial")

        self._serial = serial.Serial(
            port=self._port,
            baudrate=self._baudrate,
            timeout=0.1,
        )
        if not self._serial.is_open:
            self._serial.open()
        self._serial.reset_input_buffer()
        self._logger.info(
            "MR60BHA2 opened on %s @ %d baud", self._port, self._baudrate
        )

    async def _disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
        self._serial = None

    async def _read(self) -> SensorReading:
        """Read and parse one complete Seeed mmWave frame.

        Blocks until a valid frame is received, then updates internal
        state and returns the latest snapshot of all vital signs.
        """
        raw = await asyncio.get_event_loop().run_in_executor(
            None, self._read_frame_sync
        )
        self._parse_frame(raw)

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "heart_rate_bpm": self._heart_rate_bpm,
                "breathing_rate_bpm": self._breathing_rate,
                "distance_cm": self._distance_cm,
                "person_present": self._person_present,
            },
            raw=raw,
        )

    # -- Seeed mmWave protocol parser (ported from Seeed-mmWave-library) ------

    def _read_frame_sync(self) -> bytes:
        """Block until a complete, checksum-valid frame is received."""
        ser = self._serial
        while True:
            # Scan for SOF byte
            b = ser.read(1)
            if not b or b[0] != _SOF:
                continue

            # Read remaining 7 header bytes
            hdr_rest = ser.read(7)
            if len(hdr_rest) < 7:
                continue

            header = bytes([_SOF]) + hdr_rest
            # Verify header checksum
            xor = 0
            for i in range(7):
                xor ^= header[i]
            expected_hcs = (~xor) & 0xFF
            if header[7] != expected_hcs:
                continue

            # Parse data length
            data_len = struct.unpack(">H", header[3:5])[0]
            if data_len > _MAX_PAYLOAD:
                continue

            # Read payload + data checksum
            payload_and_cs = ser.read(data_len + 1)
            if len(payload_and_cs) < data_len + 1:
                continue

            payload = payload_and_cs[:data_len]
            data_cs = payload_and_cs[data_len]

            # Verify data checksum
            xor = 0
            for byte in payload:
                xor ^= byte
            if ((~xor) & 0xFF) != data_cs:
                continue

            return header + payload_and_cs

    def _parse_frame(self, raw: bytes) -> None:
        """Parse a validated frame and update internal state."""
        frame_type = struct.unpack(">H", raw[5:7])[0]
        data_len = struct.unpack(">H", raw[3:5])[0]
        payload = raw[8 : 8 + data_len]

        if frame_type == _TYPE_BREATHING and len(payload) >= 4:
            self._breathing_rate = struct.unpack("<f", payload[:4])[0]

        elif frame_type == _TYPE_HEART_RATE and len(payload) >= 4:
            self._heart_rate_bpm = struct.unpack("<f", payload[:4])[0]

        elif frame_type == _TYPE_DISTANCE and len(payload) >= 4:
            self._distance_cm = struct.unpack("<f", payload[:4])[0]

        elif frame_type == _TYPE_PRESENCE and len(payload) >= 1:
            self._person_present = payload[0] != 0
