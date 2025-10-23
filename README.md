# MyBuoyTools

End-to-end workflow to validate a WW3 model against buoy observations (NDBC and EMODNET), compute summary metrics, and generate global validation figures.

Outputs include:
- stations_match.csv: station coordinates and common time window
- summary_statistics.csv: per-station metrics (bias, rmse, mae, si, correlation, p99 stats, index of agreement)
- validation_map_NRMSE.png, validation_map_NBIAS.png, validation_map_CORR.png, validation_maps.png
- statistics_summary_plot.png (normalized metrics) and statistics_p99_plot.png (extremes)
- validation_metrics.csv: compact table for dashboards (station, type, lon/lat, nrmse%, nbias%, correlation, n_points, hs_obs_mean)

Data sources:
- NDBC (https://www.ndbc.noaa.gov/)
- EMODNET (https://emodnet.ec.europa.eu/en)
- PNBOIA (http://www.goosbrasil.org/pnboia/)

## Quickstart

Prereqs:
- macOS/Linux, Python 3.10+
- Buoy data folders:
	- NDBC data and corresponding WW3/model files in: ~/Downloads/ndbc
	- EMODNET data and corresponding WW3/model files in: ~/Downloads/emodnet

Setup once in this repo:

```bash
# Create and activate venv (optional if .venv already present)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install pandas numpy xarray matplotlib seaborn cartopy netCDF4 cftime
```

Run everything end-to-end (default paths point to ~/Downloads):

```bash
python scripts/run_pipeline.py \
	--emodnet "$HOME/Downloads/emodnet" \
	--ndbc "$HOME/Downloads/ndbc"
```

Key outputs will be written into `scripts/`.

## Full workflow (manual steps)

You can run each stage individually if you prefer finer control:

1) Build station matches and time intersections

```bash
python scripts/check-dados.py "$HOME/Downloads/emodnet" "$HOME/Downloads/ndbc"
```

2) Compute per-station statistics

```bash
python scripts/create_summary_stats.py
```

3) Create validation maps

```bash
python scripts/create_individual_maps.py
python scripts/create_validation_maps.py
```

4) Create statistics plots (normalized + P99)

```bash
python scripts/plot_statistics_summary.py
python scripts/plot_statistics_p99.py
```

5) Export compact CSV for dashboards

```bash
python scripts/export_metrics_csv.py
```

## Notes and assumptions

- Time handling: EMODNET time is treated as “hours since 2020-01-01”. NDBC CSV/TXT are parsed into datetimes.
- Coordinates: For NetCDF with constant lon/lat arrays, we extract the first finite value to avoid float overflow.
- Data quality filters: Observation Hs filtered to drop NaN/Inf, negative, and > 30 m values.
- Metrics:
	- Normalized: NRMSE = rmse/mean(obs)*100, NBIAS = bias/mean(obs)*100
	- Additional: MAE, SI = rmse/mean(obs), Correlation (Pearson), Index of Agreement (d)
	- Extremes: P99 metrics computed on observations ≥ 99th percentile

## Troubleshooting

- Missing pandas/numpy/xarray: re-run `pip install` above inside the venv.
- Cartopy install issues on macOS: ensure Xcode CLT and proj/geos are present (e.g., via Homebrew: `brew install proj geos`), then reinstall cartopy.
- No stations matched: verify filenames include station IDs and that both obs and model files exist in the EMODNET/NDBC folders.






