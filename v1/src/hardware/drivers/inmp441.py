"""
INMP441 MEMS microphone — audio level (SPL) and raw PCM stream.

Uses the MicroPython ``machine.I2S`` API on ESP32, or falls back to
``sounddevice`` on Linux/RPi for development and testing.

On MicroPython (ESP32):
    from machine import I2S, Pin
    — SCK, WS, SD pins configured at init

On CPython (Linux/RPi):
    pip install sounddevice numpy
    — Uses ALSA/PulseAudio I2S capture device

Hardware: INMP441 breakout wired to I2S bus (SCK, WS, SD, 3.3 V).
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, Optional, Set

import numpy as np

from v1.src.hardware.base import (
    SensorBus,
    SensorCapability,
    SensorDriver,
    SensorReading,
)

# Audio constants
_SAMPLE_RATE = 16_000   # 16 kHz — sufficient for environment sensing
_BLOCK_SIZE = 1024      # samples per read block
_CHANNELS = 1           # mono
_BIT_DEPTH = 32         # INMP441 outputs 24-bit in 32-bit I2S frame
_REF_PRESSURE = 20e-6   # reference pressure for dB SPL (20 uPa)


class INMP441Driver(SensorDriver):
    """Driver for INMP441 MEMS microphone via I2S.

    Parameters
    ----------
    sensor_id : str
        Unique identifier.
    sck_pin : int
        I2S serial clock pin (MicroPython only).
    ws_pin : int
        I2S word select pin (MicroPython only).
    sd_pin : int
        I2S serial data pin (MicroPython only).
    sample_rate : int
        Audio sample rate in Hz.
    block_size : int
        Number of samples per read block.
    device_index : int or None
        Audio device index (CPython/sounddevice only).
    """

    def __init__(
        self,
        sensor_id: str = "inmp441",
        sck_pin: int = 26,
        ws_pin: int = 25,
        sd_pin: int = 33,
        sample_rate: int = _SAMPLE_RATE,
        block_size: int = _BLOCK_SIZE,
        device_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(sensor_id, **kwargs)
        self._sck_pin = sck_pin
        self._ws_pin = ws_pin
        self._sd_pin = sd_pin
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._device_index = device_index
        self._backend: Optional[str] = None  # "micropython" or "sounddevice"
        self._i2s: Any = None
        self._stream: Any = None

    @property
    def capabilities(self) -> Set[SensorCapability]:
        return {
            SensorCapability.AUDIO_LEVEL,
            SensorCapability.AUDIO_STREAM,
        }

    @property
    def bus_type(self) -> SensorBus:
        return SensorBus.I2S

    async def _connect(self) -> None:
        # Try MicroPython first, then fall back to sounddevice
        try:
            self._connect_micropython()
            self._backend = "micropython"
            return
        except ImportError:
            pass

        try:
            self._connect_sounddevice()
            self._backend = "sounddevice"
            return
        except ImportError:
            raise ImportError(
                "INMP441 driver requires either MicroPython (machine.I2S) "
                "or CPython with: pip install sounddevice numpy"
            )

    def _connect_micropython(self) -> None:
        """Initialize MicroPython I2S peripheral."""
        from machine import I2S, Pin  # type: ignore[import-not-found]

        self._i2s = I2S(
            0,  # I2S peripheral ID
            sck=Pin(self._sck_pin),
            ws=Pin(self._ws_pin),
            sd=Pin(self._sd_pin),
            mode=I2S.RX,
            bits=_BIT_DEPTH,
            format=I2S.MONO,
            rate=self._sample_rate,
            ibuf=self._block_size * 4,  # 4 bytes per 32-bit sample
        )
        self._logger.info(
            "INMP441 MicroPython I2S: SCK=%d WS=%d SD=%d @ %d Hz",
            self._sck_pin, self._ws_pin, self._sd_pin, self._sample_rate,
        )

    def _connect_sounddevice(self) -> None:
        """Initialize sounddevice InputStream for CPython."""
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=_CHANNELS,
            dtype="int32",
            device=self._device_index,
        )
        self._stream.start()
        self._logger.info(
            "INMP441 sounddevice: device=%s @ %d Hz",
            self._device_index or "default", self._sample_rate,
        )

    async def _disconnect(self) -> None:
        if self._backend == "micropython" and self._i2s:
            self._i2s.deinit()
            self._i2s = None
        elif self._backend == "sounddevice" and self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._backend = None

    async def _read(self) -> SensorReading:
        """Read one block of audio and compute SPL."""
        samples = await asyncio.get_event_loop().run_in_executor(
            None, self._read_block_sync
        )

        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

        # Convert to approximate dB SPL
        # INMP441 sensitivity: -26 dBFS = 94 dB SPL
        # Full-scale for 24-bit in 32-bit frame: 2^23
        full_scale = 2**23
        if rms > 0:
            db_fs = 20.0 * math.log10(rms / full_scale)
            db_spl = db_fs + 94.0 + 26.0  # offset by sensitivity
        else:
            db_spl = 0.0

        peak = float(np.max(np.abs(samples)))

        return SensorReading(
            sensor_id=self.sensor_id,
            timestamp_us=self._now_us(),
            capabilities=self.capabilities,
            values={
                "db_spl": round(db_spl, 1),
                "rms": rms,
                "peak": peak,
                "sample_rate": self._sample_rate,
                "block_size": self._block_size,
            },
            raw=samples.tobytes(),
        )

    def _read_block_sync(self) -> np.ndarray:
        """Blocking read of one audio block."""
        if self._backend == "micropython":
            buf = bytearray(self._block_size * 4)
            self._i2s.readinto(buf)
            # INMP441 sends 24-bit left-justified in 32-bit I2S frame
            samples = np.frombuffer(buf, dtype=np.int32)
            # Shift right 8 to get 24-bit signed value
            return samples >> 8

        elif self._backend == "sounddevice":
            data, _ = self._stream.read(self._block_size)
            # sounddevice returns (block_size, channels) array
            return data[:, 0] if data.ndim > 1 else data

        raise RuntimeError("No audio backend connected")
