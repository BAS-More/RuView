"""
WebSocket sensing server.

Lightweight asyncio server that bridges the WiFi sensing pipeline to the
browser UI.  Runs the RSSI feature extractor + classifier on a 500 ms
tick and broadcasts JSON frames to all connected WebSocket clients on
``ws://localhost:8765``.

Usage
-----
    pip install websockets
    python -m v1.src.sensing.ws_server          # or  python v1/src/sensing/ws_server.py

Data sources (tried in order):
    1. ESP32 CSI over UDP port 5005 (ADR-018 binary frames)
    2. Windows WiFi RSSI via netsh
    3. Linux WiFi RSSI via /proc/net/wireless
    4. Simulated collector (fallback)
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import signal
import socket
import struct
import sys
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np

# Sensing pipeline imports
from v1.src.sensing.rssi_collector import (
    WifiSample,
    RingBuffer,
)
from v1.src.sensing.feature_extractor import RssiFeatureExtractor, RssiFeatures
from v1.src.sensing.classifier import MotionLevel, PresenceClassifier, SensingResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "localhost"
PORT = 8765
TICK_INTERVAL = 0.5  # seconds between broadcasts
SIGNAL_FIELD_GRID = 20  # NxN grid for signal field visualization
ESP32_UDP_PORT = 5005


# ---------------------------------------------------------------------------
# ESP32 UDP Collector — reads ADR-018 binary frames
# ---------------------------------------------------------------------------

class Esp32UdpCollector:
    """
    Collects real CSI data from ESP32 nodes via UDP (ADR-018 binary format).

    Parses I/Q pairs, computes mean amplitude per frame, and stores it as
    an RSSI-equivalent value in the standard WifiSample ring buffer so the
    existing feature extractor and classifier work unchanged.

    Also keeps the last parsed CSI frame for the UI to show subcarrier data.
    """

    # ADR-018 header: magic(4) node_id(1) n_ant(1) n_sc(2) freq(4) seq(4) rssi(1) noise(1) reserved(2)
    MAGIC = 0xC5110001
    HEADER_SIZE = 20
    HEADER_FMT = '<IBBHIIBB2x'

    def __init__(
        self,
        bind_addr: str = "0.0.0.0",
        port: int = ESP32_UDP_PORT,
        sample_rate_hz: float = 10.0,
        buffer_seconds: int = 120,
    ) -> None:
        self._bind = bind_addr
        self._port = port
        self._rate = sample_rate_hz
        self._buffer = RingBuffer(max_size=int(sample_rate_hz * buffer_seconds))
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

        # Last CSI frame for enhanced UI
        self.last_csi: Optional[Dict] = None
        self._frames_received = 0

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    @property
    def frames_received(self) -> int:
        return self._frames_received

    def start(self) -> None:
        if self._running:
            return
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self._bind, self._port))
        self._running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="esp32-udp-collector"
        )
        self._thread.start()
        logger.info("Esp32UdpCollector listening on %s:%d", self._bind, self._port)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock:
            self._sock.close()
            self._sock = None
        logger.info("Esp32UdpCollector stopped (%d frames received)", self._frames_received)

    def get_samples(self, n: Optional[int] = None) -> List[WifiSample]:
        if n is not None:
            return self._buffer.get_last_n(n)
        return self._buffer.get_all()

    def _recv_loop(self) -> None:
        while self._running:
            try:
                data, addr = self._sock.recvfrom(4096)
                self._parse_and_store(data, addr)
            except socket.timeout:
                continue
            except Exception:
                if self._running:
                    logger.exception("Error receiving ESP32 UDP packet")

    def _parse_and_store(self, raw: bytes, addr) -> None:
        if len(raw) < self.HEADER_SIZE:
            return

        magic, node_id, n_ant, n_sc, freq_mhz, seq, rssi_u8, noise_u8 = \
            struct.unpack_from(self.HEADER_FMT, raw, 0)

        if magic != self.MAGIC:
            return

        rssi = rssi_u8 if rssi_u8 < 128 else rssi_u8 - 256
        noise = noise_u8 if noise_u8 < 128 else noise_u8 - 256

        # Parse I/Q data if available
        iq_count = n_ant * n_sc
        iq_bytes_needed = self.HEADER_SIZE + iq_count * 2
        amplitude_list = []

        if len(raw) >= iq_bytes_needed and iq_count > 0:
            iq_raw = struct.unpack_from(f'<{iq_count * 2}b', raw, self.HEADER_SIZE)
            i_vals = np.array(iq_raw[0::2], dtype=np.float64)
            q_vals = np.array(iq_raw[1::2], dtype=np.float64)
            amplitudes = np.sqrt(i_vals ** 2 + q_vals ** 2)
            mean_amp = float(np.mean(amplitudes))
            amplitude_list = amplitudes.tolist()
        else:
            mean_amp = 0.0

        # Store enhanced CSI info for UI
        self.last_csi = {
            "node_id": node_id,
            "n_antennas": n_ant,
            "n_subcarriers": n_sc,
            "freq_mhz": freq_mhz,
            "sequence": seq,
            "rssi_dbm": rssi,
            "noise_floor_dbm": noise,
            "mean_amplitude": mean_amp,
            "amplitude": amplitude_list[:56],  # cap for JSON size
            "source_addr": f"{addr[0]}:{addr[1]}",
        }

        # Use RSSI from the ESP32 frame header as the primary signal metric.
        # If RSSI is the default -80 placeholder, derive a pseudo-RSSI from
        # mean amplitude to keep the feature extractor meaningful.
        effective_rssi = float(rssi)
        if rssi == -80 and mean_amp > 0:
            # Map amplitude (typically 1-20) to dBm range (-70 to -30)
            effective_rssi = -70.0 + min(mean_amp, 20.0) * 2.0

        sample = WifiSample(
            timestamp=time.time(),
            rssi_dbm=effective_rssi,
            noise_dbm=float(noise),
            link_quality=max(0.0, min(1.0, (effective_rssi + 100.0) / 60.0)),
            tx_bytes=seq * 1500,
            rx_bytes=seq * 3000,
            retry_count=0,
            interface=f"esp32-node{node_id}",
        )
        self._buffer.append(sample)
        self._frames_received += 1


# ---------------------------------------------------------------------------
# Probe for ESP32 UDP
# ---------------------------------------------------------------------------

def probe_esp32_udp(port: int = ESP32_UDP_PORT, timeout: float = 2.0) -> bool:
    """Return True if an ESP32 is actively streaming on the UDP port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(timeout)
    try:
        sock.bind(("0.0.0.0", port))
        data, _ = sock.recvfrom(256)
        if len(data) >= 20:
            magic = struct.unpack_from('<I', data, 0)[0]
            return magic == 0xC5110001
        return False
    except (socket.timeout, OSError):
        return False
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Signal field generator
# ---------------------------------------------------------------------------

def generate_signal_field(
    features: RssiFeatures,
    result: SensingResult,
    grid_size: int = SIGNAL_FIELD_GRID,
    csi_data: Optional[Dict] = None,
) -> Dict:
    """
    Generate a 2-D signal-strength field for the Gaussian splat visualization.
    When real CSI amplitude data is available, it modulates the field.
    """
    field = np.zeros((grid_size, grid_size), dtype=np.float64)

    # Base noise floor
    rng = np.random.default_rng(int(abs(features.mean * 100)) % (2**31))
    field += rng.uniform(0.02, 0.08, size=(grid_size, grid_size))

    cx, cy = grid_size // 2, grid_size // 2

    # Radial attenuation from router
    for y in range(grid_size):
        for x in range(grid_size):
            dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            attenuation = max(0.0, 1.0 - dist / (grid_size * 0.7))
            field[y, x] += attenuation * 0.3

    # If we have real CSI subcarrier amplitudes, paint them along one axis
    if csi_data and csi_data.get("amplitude"):
        amps = np.array(csi_data["amplitude"][:grid_size], dtype=np.float64)
        if len(amps) > 0:
            max_a = np.max(amps) if np.max(amps) > 0 else 1.0
            norm_amps = amps / max_a
            # Spread subcarrier energy as vertical stripes
            for ix, a in enumerate(norm_amps):
                col = int(ix * grid_size / len(norm_amps))
                col = min(col, grid_size - 1)
                field[:, col] += a * 0.4

    if result.presence_detected:
        body_x = cx + int(3 * math.sin(time.time() * 0.2))
        body_y = cy + int(2 * math.cos(time.time() * 0.15))
        sigma = 2.0 + features.variance * 0.5

        for y in range(grid_size):
            for x in range(grid_size):
                dx = x - body_x
                dy = y - body_y
                blob = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
                intensity = 0.3 + 0.7 * min(1.0, features.motion_band_power * 5)
                field[y, x] += blob * intensity

        if features.breathing_band_power > 0.01:
            breath_phase = math.sin(2 * math.pi * 0.3 * time.time())
            breath_radius = 3.0 + breath_phase * 0.8
            for y in range(grid_size):
                for x in range(grid_size):
                    dist_body = math.sqrt((x - body_x) ** 2 + (y - body_y) ** 2)
                    ring = math.exp(-((dist_body - breath_radius) ** 2) / 1.5)
                    field[y, x] += ring * features.breathing_band_power * 2

    field = np.clip(field, 0.0, 1.0)

    return {
        "grid_size": [grid_size, 1, grid_size],
        "values": field.flatten().tolist(),
    }


# ---------------------------------------------------------------------------
# WebSocket server
# ---------------------------------------------------------------------------

class SensingWebSocketServer:
    """Async WebSocket server that broadcasts sensing updates.

    When Phase A sensors are available, uses ``MultiSensorBackend`` to
    fuse WiFi CSI/RSSI with sensor data into a ``FusedSensingResult``
    that is broadcast to all connected WebSocket clients.
    """

    def __init__(self) -> None:
        self.clients: Set = set()
        self.collector = None
        self.extractor = RssiFeatureExtractor(window_seconds=10.0)
        self.classifier = PresenceClassifier()
        self.source: str = "unknown"
        self._running = False

        # Phase A multi-modal sensor registry (lazy-initialised)
        self.sensor_registry = None
        self._fusion_backend = None  # MultiSensorBackend when sensors available
        self._last_fused = None      # Last FusedSensingResult
        self._use_simulated_sensors = False  # set via --simulate-sensors flag

        # Recording and playback
        self._recorder = None        # SensorRecorder when --record is active
        self._playback_path = None   # Path for --playback mode

    def _create_collector(self):
        """Auto-detect data source: ESP32 UDP > platform WiFi > simulated.

        Uses the ``create_collector`` factory (ADR-049) for platform WiFi
        detection, which never raises and logs actionable fallback messages.
        """
        from .rssi_collector import create_collector

        # 1. Try ESP32 UDP first
        print("  Probing for ESP32 on UDP :5005 ...")
        if probe_esp32_udp(ESP32_UDP_PORT, timeout=2.0):
            logger.info("ESP32 CSI stream detected on UDP :%d", ESP32_UDP_PORT)
            self.source = "esp32"
            return Esp32UdpCollector(port=ESP32_UDP_PORT, sample_rate_hz=10.0)

        # 2. Platform-specific WiFi (auto-detect with graceful fallback)
        collector = create_collector(preferred="auto", sample_rate_hz=10.0)

        # Map collector class to source label
        source_map = {
            "LinuxWifiCollector": "linux_wifi",
            "WindowsWifiCollector": "windows_wifi",
            "MacosWifiCollector": "macos_wifi",
            "SimulatedCollector": "simulated",
        }
        self.source = source_map.get(type(collector).__name__, "unknown")
        return collector

    def _build_message(self, features: RssiFeatures, result: SensingResult) -> str:
        """Build the JSON message to broadcast."""
        # Get CSI-specific data if available
        csi_data = None
        if isinstance(self.collector, Esp32UdpCollector):
            csi_data = self.collector.last_csi

        signal_field = generate_signal_field(features, result, csi_data=csi_data)

        node_info = {
            "node_id": 1,
            "rssi_dbm": features.mean,
            "position": [2.0, 0.0, 1.5],
            "amplitude": [],
            "subcarrier_count": 0,
        }

        # Enrich with real CSI data
        if csi_data:
            node_info["node_id"] = csi_data.get("node_id", 1)
            node_info["rssi_dbm"] = csi_data.get("rssi_dbm", features.mean)
            node_info["amplitude"] = csi_data.get("amplitude", [])
            node_info["subcarrier_count"] = csi_data.get("n_subcarriers", 0)
            node_info["mean_amplitude"] = csi_data.get("mean_amplitude", 0)
            node_info["freq_mhz"] = csi_data.get("freq_mhz", 0)
            node_info["sequence"] = csi_data.get("sequence", 0)
            node_info["source_addr"] = csi_data.get("source_addr", "")

        msg = {
            "type": "sensing_update",
            "timestamp": time.time(),
            "source": self.source,
            "nodes": [node_info],
            "features": {
                "mean_rssi": features.mean,
                "variance": features.variance,
                "std": features.std,
                "motion_band_power": features.motion_band_power,
                "breathing_band_power": features.breathing_band_power,
                "dominant_freq_hz": features.dominant_freq_hz,
                "change_points": features.n_change_points,
                "spectral_power": features.total_spectral_power,
                "range": features.range,
                "iqr": features.iqr,
                "skewness": features.skewness,
                "kurtosis": features.kurtosis,
            },
            "classification": {
                "motion_level": result.motion_level.value,
                "presence": result.presence_detected,
                "confidence": round(result.confidence, 3),
            },
            "signal_field": signal_field,
        }

        # Merge fused multi-sensor data if available
        if self._last_fused is not None:
            fused = self._last_fused
            msg["fusion"] = {
                "presence": fused.presence,
                "presence_sources": fused.presence_sources,
                "fused_confidence": round(fused.fused_confidence, 3),
            }
            if fused.heart_rate_bpm is not None:
                msg["fusion"]["heart_rate_bpm"] = fused.heart_rate_bpm
            if fused.breathing_rate_bpm is not None:
                msg["fusion"]["breathing_rate_bpm"] = fused.breathing_rate_bpm
            if fused.nearest_distance_mm is not None:
                msg["fusion"]["nearest_distance_mm"] = fused.nearest_distance_mm
                msg["fusion"]["target_count"] = fused.target_count
            if fused.temperature_c is not None:
                msg["fusion"]["environment"] = {
                    "temperature_c": fused.temperature_c,
                    "humidity_pct": fused.humidity_pct,
                    "pressure_hpa": fused.pressure_hpa,
                }
            if fused.tvoc_ppb is not None:
                msg["fusion"]["air_quality"] = {
                    "tvoc_ppb": fused.tvoc_ppb,
                    "eco2_ppm": fused.eco2_ppm,
                    "aqi": fused.aqi,
                }
            if fused.thermal_max_c is not None:
                msg["fusion"]["thermal"] = {
                    "max_c": fused.thermal_max_c,
                    "presence": fused.thermal_presence,
                }
            if fused.db_spl is not None:
                msg["fusion"]["audio"] = {"db_spl": fused.db_spl}
            if fused.sensor_readings:
                msg["sensors"] = fused.sensor_readings

        return json.dumps(msg)

    async def _handler(self, websocket):
        """Handle a single WebSocket client connection."""
        self.clients.add(websocket)
        remote = websocket.remote_address
        logger.info("Client connected: %s", remote)
        try:
            async for _ in websocket:
                pass
        finally:
            self.clients.discard(websocket)
            logger.info("Client disconnected: %s", remote)

    async def _broadcast(self, message: str) -> None:
        """Send message to all connected clients."""
        if not self.clients:
            return
        disconnected = set()
        for ws in self.clients:
            try:
                await ws.send(message)
            except Exception:
                disconnected.add(ws)
        self.clients -= disconnected

    async def _tick_loop(self) -> None:
        """Main sensing loop."""
        while self._running:
            try:
                window = self.extractor.window_seconds
                sample_rate = self.collector.sample_rate_hz
                n_needed = int(window * sample_rate)
                samples = self.collector.get_samples(n=n_needed)

                if len(samples) >= 4:
                    features = self.extractor.extract(samples)
                    result = self.classifier.classify(features)

                    # Run fusion if multi-sensor backend is available
                    if self._fusion_backend:
                        try:
                            self._last_fused = await self._fusion_backend.fuse()
                            # Record the fused frame if recording is active
                            if self._recorder and self._recorder.is_recording:
                                self._recorder.record_frame(self._last_fused)
                        except Exception as exc:
                            logger.debug("Fusion cycle error: %s", exc)

                    message = self._build_message(features, result)
                    await self._broadcast(message)

                    # Print status every few ticks
                    if isinstance(self.collector, Esp32UdpCollector):
                        csi = self.collector.last_csi
                        if csi and self.collector.frames_received % 20 == 0:
                            print(
                                f"  [{csi['source_addr']}] node:{csi['node_id']} "
                                f"seq:{csi['sequence']} sc:{csi['n_subcarriers']} "
                                f"rssi:{csi['rssi_dbm']}dBm amp:{csi['mean_amplitude']:.1f} "
                                f"=> {result.motion_level.value} ({result.confidence:.0%})"
                            )
                else:
                    logger.debug("Waiting for samples (%d/%d)", len(samples), n_needed)
            except Exception:
                logger.exception("Error in sensing tick")

            await asyncio.sleep(TICK_INTERVAL)

    async def _init_sensor_registry(self) -> None:
        """Probe Phase A sensors and create MultiSensorBackend.

        Falls back to simulated sensors when ``--simulate-sensors`` is
        active or no real hardware responds.
        """
        try:
            from v1.src.hardware.sensor_registry import SensorRegistry
            from v1.src.sensing.multi_sensor_backend import MultiSensorBackend
            from v1.src.sensing.backend import CommodityBackend

            self.sensor_registry = SensorRegistry()
            detected = await self.sensor_registry.auto_detect()

            # Fall back to simulated sensors if none detected
            if not detected and self._use_simulated_sensors:
                from v1.src.hardware.drivers.simulated import SimulatedSensorSuite
                suite = SimulatedSensorSuite()
                self.sensor_registry = await suite.create_registry()
                detected = list(self.sensor_registry.sensors.keys())
                print(f"  Phase A sensors (SIMULATED): {', '.join(detected)}")

            if detected:
                wifi_backend = CommodityBackend(
                    self.collector, self.extractor, self.classifier
                )
                self._fusion_backend = MultiSensorBackend(
                    wifi_backend, self.sensor_registry
                )
                caps = sorted(c.name for c in self._fusion_backend.get_capabilities())
                if not self._use_simulated_sensors:
                    print(f"  Phase A sensors: {', '.join(detected)}")
                print(f"  Fusion capabilities: {', '.join(caps)}")
            else:
                print("  Phase A sensors: none detected (WiFi-only mode)")
        except Exception as exc:
            logger.debug("Phase A sensor init skipped: %s", exc)
            self.sensor_registry = None

    async def run(self) -> None:
        """Start the server and run until interrupted."""
        try:
            import websockets
        except ImportError:
            print("ERROR: 'websockets' package not found.")
            print("Install it with:  pip install websockets")
            sys.exit(1)

        self.collector = self._create_collector()
        self.collector.start()
        self._running = True

        # Probe Phase A sensors (non-blocking, failures are fine)
        await self._init_sensor_registry()

        print(f"\n  Sensing WebSocket server on ws://{HOST}:{PORT}")
        print(f"  Source: {self.source}")
        print(f"  Tick: {TICK_INTERVAL}s | Window: {self.extractor.window_seconds}s")
        print("  Press Ctrl+C to stop\n")

        async with websockets.serve(self._handler, HOST, PORT):
            if self._playback_path:
                await self._playback_loop()
            else:
                await self._tick_loop()

    async def _playback_loop(self) -> None:
        """Replay a recorded JSONL session to all WebSocket clients."""
        from v1.src.sensing.recorder import SensorPlayer

        player = SensorPlayer(self._playback_path)
        n = player.load()
        print(f"  Replaying {n} frames from {self._playback_path}")

        async for frame in player.play(speed=1.0):
            if not self._running:
                break
            fused = player.as_fused_result(frame)
            self._last_fused = fused

            # Reconstruct minimal features from the WiFi result
            features = RssiFeatures(
                mean=-55.0,
                variance=fused.wifi.rssi_variance,
                std=fused.wifi.rssi_variance ** 0.5,
                range=5.0,
                iqr=2.5,
                skewness=0.0,
                kurtosis=3.0,
                motion_band_power=fused.wifi.motion_band_energy,
                breathing_band_power=fused.wifi.breathing_band_energy,
                total_spectral_power=0.5,
                dominant_freq_hz=0.5,
                n_change_points=fused.wifi.n_change_points,
            )
            result = SensingResult(
                motion_level=fused.wifi.motion_level,
                confidence=fused.wifi.confidence,
                presence_detected=fused.wifi.presence_detected,
                rssi_variance=fused.wifi.rssi_variance,
                motion_band_energy=fused.wifi.motion_band_energy,
                breathing_band_energy=fused.wifi.breathing_band_energy,
                n_change_points=fused.wifi.n_change_points,
            )

            message = self._build_message(features, result)
            await self._broadcast(message)

        print("  Playback complete")

    def stop(self) -> None:
        """Stop the server gracefully."""
        self._running = False
        if self._recorder and self._recorder.is_recording:
            self._recorder.stop()
        if self.collector:
            self.collector.stop()
        # Shutdown Phase A sensors
        if self.sensor_registry:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.sensor_registry.shutdown())
                else:
                    loop.run_until_complete(self.sensor_registry.shutdown())
            except RuntimeError:
                pass
        logger.info("Sensing server stopped")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="RuView Sensing WebSocket Server")
    parser.add_argument(
        "--simulate-sensors",
        action="store_true",
        help="Use simulated Phase A sensors (no hardware needed)",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="Record fused sensor data to a JSONL file",
    )
    parser.add_argument(
        "--playback",
        metavar="PATH",
        help="Replay a recorded JSONL session instead of live sensing",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    server = SensingWebSocketServer()
    server._use_simulated_sensors = args.simulate_sensors

    if args.record:
        from v1.src.sensing.recorder import SensorRecorder
        server._recorder = SensorRecorder(args.record)
        server._recorder.start()
        print(f"  Recording to: {args.record}")

    if args.playback:
        server._playback_path = args.playback
        # Force simulated sensors for playback (no hardware needed)
        server._use_simulated_sensors = True

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(sig, frame):
        print("\nShutting down...")
        server.stop()
        loop.stop()

    signal.signal(signal.SIGINT, _shutdown)

    try:
        loop.run_until_complete(server.run())
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        loop.close()


if __name__ == "__main__":
    main()
