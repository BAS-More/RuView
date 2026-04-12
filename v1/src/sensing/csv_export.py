"""
Export fused sensor recordings to CSV for analysis.

Reads a JSONL recording file and writes a flat CSV with one row per
frame — useful for importing into Excel, pandas, or Jupyter.

Usage::

    exporter = CsvExporter("data/recordings/session.jsonl")
    exporter.export("data/exports/session.csv")
    # or from CLI: python -m v1.src.cli sensor export session.jsonl -o session.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


# Columns in output order
COLUMNS = [
    "frame_id",
    "timestamp",
    "presence",
    "fused_confidence",
    "presence_sources",
    "motion_level",
    "wifi_confidence",
    "heart_rate_bpm",
    "breathing_rate_bpm",
    "nearest_distance_mm",
    "target_count",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "tvoc_ppb",
    "eco2_ppm",
    "aqi",
    "thermal_max_c",
    "thermal_presence",
    "db_spl",
]


class CsvExporter:
    """Export JSONL sensor recordings to flat CSV.

    Parameters
    ----------
    input_path : str or Path
        Path to the JSONL recording file.
    """

    def __init__(self, input_path: str | Path) -> None:
        self._input = Path(input_path)

    def export(self, output_path: str | Path, columns: Optional[List[str]] = None) -> int:
        """Export to CSV.

        Parameters
        ----------
        output_path : str or Path
            Path to the output CSV file.
        columns : list of str, optional
            Columns to include. Defaults to ``COLUMNS``.

        Returns
        -------
        int
            Number of rows written.
        """
        cols = columns or COLUMNS
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        rows_written = 0
        with open(self._input, "r", encoding="utf-8") as fin, \
             open(output, "w", newline="", encoding="utf-8") as fout:
            writer = csv.DictWriter(fout, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()

            for line in fin:
                line = line.strip()
                if not line:
                    continue
                frame = json.loads(line)
                row = self._flatten(frame)
                writer.writerow(row)
                rows_written += 1

        return rows_written

    @staticmethod
    def _flatten(frame: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a JSONL frame into a flat dict for CSV."""
        f = frame.get("fusion", {})
        w = frame.get("wifi", {})
        return {
            "frame_id": frame.get("frame_id", ""),
            "timestamp": frame.get("timestamp", ""),
            "presence": f.get("presence", ""),
            "fused_confidence": f.get("fused_confidence", ""),
            "presence_sources": ";".join(f.get("presence_sources", [])),
            "motion_level": w.get("motion_level", ""),
            "wifi_confidence": w.get("confidence", ""),
            "heart_rate_bpm": f.get("heart_rate_bpm", ""),
            "breathing_rate_bpm": f.get("breathing_rate_bpm", ""),
            "nearest_distance_mm": f.get("nearest_distance_mm", ""),
            "target_count": f.get("target_count", ""),
            "temperature_c": f.get("temperature_c", ""),
            "humidity_pct": f.get("humidity_pct", ""),
            "pressure_hpa": f.get("pressure_hpa", ""),
            "tvoc_ppb": f.get("tvoc_ppb", ""),
            "eco2_ppm": f.get("eco2_ppm", ""),
            "aqi": f.get("aqi", ""),
            "thermal_max_c": f.get("thermal_max_c", ""),
            "thermal_presence": f.get("thermal_presence", ""),
            "db_spl": f.get("db_spl", ""),
        }
