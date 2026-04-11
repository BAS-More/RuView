#!/usr/bin/env python3
"""
WiFi Sensing Research Report Generator — agentic-reports integration for RuView.

Generates weekly research reports on WiFi sensing papers, competitors,
and technology developments using the agentic-reports API.

Usage:
    python scripts/research-wifi-sensing.py
    python scripts/research-wifi-sensing.py --topic "WiFi CSI pose estimation"
    python scripts/research-wifi-sensing.py --output docs/research/

Requires:
    - agentic-reports running: pip install agentic-reports && uvicorn app.main:app
    - EXA_API_KEY and OPENAI_API_KEY environment variables
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPORTS_API = os.environ.get("AGENTIC_REPORTS_URL", "http://localhost:8000")

RESEARCH_TOPICS = [
    "WiFi CSI-based human pose estimation recent advances 2025-2026",
    "WiFi sensing vital sign monitoring non-contact health",
    "Dense pose estimation from RF signals indoor sensing",
    "WiFi-based presence detection smart home occupancy",
    "Channel State Information deep learning body tracking",
]


def generate_report(topic: str, output_dir: str) -> str | None:
    """Generate a research report for a single topic."""
    import requests

    print(f"\n  Generating report for: {topic}")

    try:
        response = requests.post(
            f"{REPORTS_API}/generate-report-advanced",
            json={
                "query": topic,
                "report_type": "research_report",
                "max_results": 10,
                "include_domains": [
                    "arxiv.org",
                    "ieee.org",
                    "acm.org",
                    "scholar.google.com",
                    "nature.com",
                    "sciencedirect.com",
                ],
                "similar_url": "https://arxiv.org/abs/2301.00250",  # DensePose from WiFi paper
            },
            timeout=120,
        )

        if response.status_code != 200:
            print(f"  ERROR: API returned {response.status_code}")
            return None

        data = response.json()
        report_content = data.get("report", "No report content")

        # Save report
        safe_topic = topic[:50].replace(" ", "-").replace("/", "-")
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"{timestamp}-{safe_topic}.md"
        filepath = Path(output_dir) / filename

        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(report_content, encoding="utf-8")
        print(f"  Saved: {filepath}")
        return str(filepath)

    except requests.exceptions.ConnectionError:
        print(f"  ERROR: Cannot connect to agentic-reports at {REPORTS_API}")
        print(f"  Start it with: uvicorn app.main:app")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate WiFi sensing research reports")
    parser.add_argument("--topic", type=str, help="Custom research topic (overrides defaults)")
    parser.add_argument("--output", type=str, default="docs/research", help="Output directory")
    parser.add_argument("--all", action="store_true", help="Generate reports for all default topics")
    args = parser.parse_args()

    print("=" * 60)
    print("RuView WiFi Sensing Research Report Generator")
    print("=" * 60)

    topics = [args.topic] if args.topic else RESEARCH_TOPICS if args.all else RESEARCH_TOPICS[:1]

    generated = []
    for topic in topics:
        result = generate_report(topic, args.output)
        if result:
            generated.append(result)

    print(f"\n{'=' * 60}")
    print(f"Generated {len(generated)}/{len(topics)} reports")
    if generated:
        print("Reports:")
        for r in generated:
            print(f"  - {r}")
    print("=" * 60)


if __name__ == "__main__":
    main()
