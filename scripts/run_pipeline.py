#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the full buoy validation workflow end-to-end.

Stages:
 1) Generate station matches (time intersections) -> stations_match.csv
 2) Compute summary statistics -> summary_statistics.csv
 3) Create validation maps (individual + combined) -> PNGs
 4) Create statistics plots (summary + P99) -> PNGs
 5) Export compact validation metrics -> validation_metrics.csv

Usage:
  python scripts/run_pipeline.py \
      --emodnet ~/Downloads/emodnet \
      --ndbc    ~/Downloads/ndbc

You can skip stages with flags like --skip-matches, --skip-stats, etc.
"""

from pathlib import Path
import argparse
import subprocess
import sys

SCRIPT_DIR = Path(__file__).parent


def run(cmd: list[str]):
    print("\n" + "=" * 70)
    print("Running:", " ".join(cmd))
    print("=" * 70)
    res = subprocess.run(cmd, cwd=SCRIPT_DIR.parent, text=True)
    if res.returncode != 0:
        print(f"❌ Command failed with code {res.returncode}")
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run buoy validation pipeline")
    parser.add_argument("--emodnet", type=str, default=str(Path.home() / "Downloads" / "emodnet"))
    parser.add_argument("--ndbc", type=str, default=str(Path.home() / "Downloads" / "ndbc"))
    parser.add_argument("--skip-matches", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--skip-maps", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    py = str((SCRIPT_DIR.parent / ".venv" / "bin" / "python").resolve())
    if not Path(py).exists():
        # Fallback to system python
        py = sys.executable

    # 1) Matches
    if not args.skip_matches:
        run([py, "scripts/check-dados.py", args.emodnet, args.ndbc])

    # 2) Stats
    if not args.skip_stats:
        run([py, "scripts/create_summary_stats.py", "--emodnet", args.emodnet, "--ndbc", args.ndbc])

    # 3) Maps
    if not args.skip_maps:
        run([py, "scripts/create_individual_maps.py"])
        run([py, "scripts/create_validation_maps.py"])

    # 4) Plots
    if not args.skip_plots:
        run([py, "scripts/plot_statistics_summary.py"])
        run([py, "scripts/plot_statistics_p99.py"])

    # 5) Export CSV
    if not args.skip_export:
        run([py, "scripts/export_metrics_csv.py"])

    print("\n" + "=" * 70)
    print("✅ Pipeline finished")
    print("Outputs in:", SCRIPT_DIR)
    print("=" * 70)


if __name__ == "__main__":
    main()
