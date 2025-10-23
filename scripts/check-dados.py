import os
import re
import sys
import csv
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
import xarray as xr

#!/usr/bin/env python3
"""
check-dados.py

Varre diretórios com dados de boias (observações) e modelo WW3, identifica estações,
compara disponibilidade temporal e salva CSV com:
estacao, lon_obs, lat_obs, lon_mod, lat_mod, tempo_inicial, tempo_final

Assume:
- Arquivos podem ser NetCDF (.nc) ou texto/CSV (.csv, .txt)
- Arquivos de modelo têm no nome 'ww3' ou 'model'
- ID da estação inferido do nome do arquivo (primeiro token alfanumérico antes de '_' ou '.')
- Variáveis de tempo: 'time', 'Times', 'timestamp'
- Variáveis de coordenadas: lon: ('lon','longitude'); lat: ('lat','latitude')
"""

try:
    import xarray as xr
except ImportError:
    xr = None

ROOT_DIRS = [
    "/Users/jtakeo/Downloads/emodnet",
    "/Users/jtakeo/Downloads/ndbc",
]

OUTPUT_CSV = "stations_match.csv"

TIME_VARS = ("time", "Times", "timestamp")
LON_VARS = ("lon", "longitude", "lon_rho")
LAT_VARS = ("lat", "latitude", "lat_rho")

MODEL_PAT = re.compile(r"(ww3|model)", re.IGNORECASE)

STATION_ID_REGEX = re.compile(r"([A-Za-z0-9]+)[_.]")


def guess_station_id(path: str) -> Optional[str]:
    """
    Extract station ID from filename.
    Tries to identify numeric station IDs (e.g., 41001) from filenames.
    For model files like ww3_41001.nc, extracts the station ID after prefix.
    """
    fname = os.path.basename(path)
    
    # Remove file extension
    name_no_ext = os.path.splitext(fname)[0]
    
    # Check if it's a model file with prefix (e.g., ww3_41001, model_41001)
    if MODEL_PAT.search(name_no_ext):
        # Try to extract station ID after model prefix
        # Pattern: ww3_41001, model_station, etc.
        parts = re.split(r'[_-]', name_no_ext)
        for part in parts[1:]:  # Skip first part (the model prefix)
            if part and re.match(r'^[A-Za-z0-9]+$', part):
                return part.lower()
    
    # For non-model files, try to extract first alphanumeric token
    m = STATION_ID_REGEX.match(fname)
    if m:
        return m.group(1).lower()
    
    return None


def is_model_file(path: str) -> bool:
    return bool(MODEL_PAT.search(os.path.basename(path)))


def list_data_files(root_dirs: List[str]) -> List[str]:
    exts = (".nc", ".csv", ".txt")
    files = []
    for rd in root_dirs:
        if not os.path.isdir(rd):
            continue
        for base, _, fns in os.walk(rd):
            for f in fns:
                if f.lower().endswith(exts):
                    files.append(os.path.join(base, f))
    return files


def extract_time_range_xr(ds) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    for tv in TIME_VARS:
        if tv in ds.variables:
            tvar = ds[tv]
            try:
                # Check if time is numeric (EMODNET format: hours since 2020-01-01)
                if tvar.dtype in [np.float32, np.float64, np.int32, np.int64]:
                    # Assume it's hours since 2020-01-01 (EMODNET format)
                    # 8784 hours = 366 days (2020 is a leap year)
                    times = pd.to_datetime('2020-01-01') + pd.to_timedelta(tvar.values, unit='h')
                else:
                    times = pd.to_datetime(tvar.values)
            except Exception:
                continue
            if times.size == 0:
                continue
            return pd.to_datetime(times.min()), pd.to_datetime(times.max())
    return None


def extract_time_range_table(df: pd.DataFrame) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    for tv in TIME_VARS:
        if tv in df.columns:
            try:
                times = pd.to_datetime(df[tv], errors="coerce")
            except Exception:
                continue
            times = times.dropna()
            if times.empty:
                continue
            return times.min(), times.max()
    # try index if datetime-like
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        return df.index.min(), df.index.max()
    return None


def extract_coords_xr(ds) -> Tuple[Optional[float], Optional[float]]:
    lon = None
    lat = None
    
    # Try to get from variables
    for lv in LON_VARS:
        if lv in ds.variables:
            val = ds[lv].values
            # Find first valid (finite) value
            if val.size > 0:
                # Flatten and find first finite value
                val_flat = val.flatten()
                finite_mask = np.isfinite(val_flat)
                if np.any(finite_mask):
                    first_valid = val_flat[finite_mask][0]
                    lon = float(first_valid)
            if lon is not None:
                break
    
    for lv in LAT_VARS:
        if lv in ds.variables:
            val = ds[lv].values
            # Find first valid (finite) value
            if val.size > 0:
                # Flatten and find first finite value
                val_flat = val.flatten()
                finite_mask = np.isfinite(val_flat)
                if np.any(finite_mask):
                    first_valid = val_flat[finite_mask][0]
                    lat = float(first_valid)
            if lat is not None:
                break
    
    # If still None, try global attributes
    if lon is None and 'longitude' in ds.attrs:
        try:
            lon = float(ds.attrs['longitude'])
        except:
            pass
    if lat is None and 'latitude' in ds.attrs:
        try:
            lat = float(ds.attrs['latitude'])
        except:
            pass
    
    return lon, lat


def extract_coords_table(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    lon = lat = None
    for c in df.columns:
        if c.lower() in LON_VARS:
            try:
                lon = float(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
            except Exception:
                pass
        if c.lower() in LAT_VARS:
            try:
                lat = float(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
            except Exception:
                pass
    return lon, lat


def read_file_metadata(path: str) -> Optional[Dict]:
    ext = os.path.splitext(path)[1].lower()
    station = guess_station_id(path)
    if not station:
        return None

    lon = lat = None
    tmin = tmax = None

    try:
        if ext == ".nc":
            if xr is None:
                return None
            with xr.open_dataset(path) as ds:
                tr = extract_time_range_xr(ds)
                if tr:
                    tmin, tmax = tr
                lon, lat = extract_coords_xr(ds)
        else:
            # CSV or TXT
            # Try comma, space, tab separated
            for sep in (",", None, r"\s+"):
                try:
                    df = pd.read_csv(path, sep=sep, engine="python")
                    break
                except Exception:
                    df = None
            if df is None:
                return None
            tr = extract_time_range_table(df)
            if tr:
                tmin, tmax = tr
            lon, lat = extract_coords_table(df)
    except Exception:
        return None

    if tmin is None or tmax is None:
        return None

    return {
        "station": station,
        "file": path,
        "lon": lon,
        "lat": lat,
        "tmin": pd.to_datetime(tmin),
        "tmax": pd.to_datetime(tmax),
        "is_model": is_model_file(path),
    }


def build_index(files: List[str]):
    obs = {}
    model = {}
    for f in files:
        md = read_file_metadata(f)
        if not md:
            continue
        dct = model if md["is_model"] else obs
        # Keep widest coverage if duplicate
        sid = md["station"]
        if sid in dct:
            old = dct[sid]
            if (md["tmin"] <= old["tmin"] and md["tmax"] >= old["tmax"]):
                dct[sid] = md
        else:
            dct[sid] = md
    return obs, model


def find_common_time_periods(file_obs: str, file_mod: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Find the actual common time periods between two files by loading their time data.
    Only considers times where BOTH files have valid (non-NaN) data.
    Returns (first_common_time, last_common_time) or None if no common times found.
    """
    try:
        # Read observation file times with valid data
        obs_times = None
        ext_obs = os.path.splitext(file_obs)[1].lower()
        
        if ext_obs == ".nc":
            if xr is None:
                return None
            with xr.open_dataset(file_obs) as ds:
                # Find time variable
                time_var = None
                for tv in TIME_VARS:
                    if tv in ds.variables:
                        time_var = tv
                        break
                
                if time_var is None:
                    return None
                
                # Find a data variable (not time or coordinates)
                data_var = None
                for var in ds.variables:
                    if var not in TIME_VARS and var not in LON_VARS and var not in LAT_VARS:
                        data_var = var
                        break
                
                if data_var is None:
                    return None
                
                try:
                    # Get times and data
                    time_values = ds[time_var].values
                    # Check if time is numeric (EMODNET format: hours since 2020-01-01)
                    if ds[time_var].dtype in [np.float32, np.float64, np.int32, np.int64]:
                        times = pd.to_datetime('2020-01-01') + pd.to_timedelta(time_values, unit='h')
                    else:
                        times = pd.to_datetime(time_values)
                    data = ds[data_var].values
                    
                    # Filter to only times with valid (non-NaN) data
                    if data.ndim > 1:
                        # If multidimensional, check if any value at each time is valid
                        valid_mask = ~pd.isna(data).all(axis=tuple(range(1, data.ndim)))
                    else:
                        valid_mask = ~pd.isna(data)
                    
                    obs_times = pd.Series(times[valid_mask]).sort_values()
                except Exception:
                    return None
        else:
            # Try reading as CSV/TXT
            for sep in (",", None, r"\s+"):
                try:
                    df = pd.read_csv(file_obs, sep=sep, engine="python")
                    time_col = None
                    for tv in TIME_VARS:
                        if tv in df.columns:
                            time_col = tv
                            break
                    
                    if time_col:
                        # Find a data column (not time)
                        data_cols = [c for c in df.columns if c not in TIME_VARS]
                        if data_cols:
                            try:
                                times = pd.to_datetime(df[time_col], errors="coerce")
                                # Filter to rows with at least one valid data value
                                valid_mask = df[data_cols].notna().any(axis=1)
                                obs_times = times[valid_mask].dropna().sort_values()
                                break
                            except Exception:
                                continue
                except Exception:
                    continue
        
        # Read model file times with valid data
        mod_times = None
        ext_mod = os.path.splitext(file_mod)[1].lower()
        
        if ext_mod == ".nc":
            if xr is None:
                return None
            with xr.open_dataset(file_mod) as ds:
                # Find time variable
                time_var = None
                for tv in TIME_VARS:
                    if tv in ds.variables:
                        time_var = tv
                        break
                
                if time_var is None:
                    return None
                
                # Find a data variable (not time or coordinates)
                data_var = None
                for var in ds.variables:
                    if var not in TIME_VARS and var not in LON_VARS and var not in LAT_VARS:
                        data_var = var
                        break
                
                if data_var is None:
                    return None
                
                try:
                    # Get times and data
                    time_values = ds[time_var].values
                    # Check if time is numeric (EMODNET format: hours since 2020-01-01)
                    if ds[time_var].dtype in [np.float32, np.float64, np.int32, np.int64]:
                        times = pd.to_datetime('2020-01-01') + pd.to_timedelta(time_values, unit='h')
                    else:
                        times = pd.to_datetime(time_values)
                    data = ds[data_var].values
                    
                    # Filter to only times with valid (non-NaN) data
                    if data.ndim > 1:
                        # If multidimensional, check if any value at each time is valid
                        valid_mask = ~pd.isna(data).all(axis=tuple(range(1, data.ndim)))
                    else:
                        valid_mask = ~pd.isna(data)
                    
                    mod_times = pd.Series(times[valid_mask]).sort_values()
                except Exception:
                    return None
        else:
            # Try reading as CSV/TXT
            for sep in (",", None, r"\s+"):
                try:
                    df = pd.read_csv(file_mod, sep=sep, engine="python")
                    time_col = None
                    for tv in TIME_VARS:
                        if tv in df.columns:
                            time_col = tv
                            break
                    
                    if time_col:
                        # Find a data column (not time)
                        data_cols = [c for c in df.columns if c not in TIME_VARS]
                        if data_cols:
                            try:
                                times = pd.to_datetime(df[time_col], errors="coerce")
                                # Filter to rows with at least one valid data value
                                valid_mask = df[data_cols].notna().any(axis=1)
                                mod_times = times[valid_mask].dropna().sort_values()
                                break
                            except Exception:
                                continue
                except Exception:
                    continue
        
        if obs_times is None or mod_times is None or len(obs_times) == 0 or len(mod_times) == 0:
            return None
        
        # Find common times (intersection)
        # Convert to sets for intersection (more efficient)
        obs_set = set(obs_times)
        mod_set = set(mod_times)
        common = obs_set & mod_set
        
        if not common:
            return None
        
        # Sort common times and get first and last
        common_sorted = sorted(common)
        return pd.Timestamp(common_sorted[0]), pd.Timestamp(common_sorted[-1])
        
    except Exception as e:
        print(f"Error finding common times: {e}")
        return None


def main():
    files = list_data_files(ROOT_DIRS)
    obs_idx, mod_idx = build_index(files)

    rows = []
    for sid, o in obs_idx.items():
        if sid not in mod_idx:
            continue
        m = mod_idx[sid]
        
        # First check if periods overlap
        start_overlap = max(o["tmin"], m["tmin"])
        end_overlap = min(o["tmax"], m["tmax"])
        
        if start_overlap > end_overlap:
            continue  # No overlap at all
        
        # Now find actual common time periods
        common_period = find_common_time_periods(o["file"], m["file"])
        
        if common_period is None:
            # If we can't determine exact common times, use overlap as before
            start = start_overlap
            end = end_overlap
        else:
            start, end = common_period
        
        rows.append({
            "estacao": sid,
            "lon_obs": o["lon"],
            "lat_obs": o["lat"],
            "lon_mod": m["lon"],
            "lat_mod": m["lat"],
            "tempo_inicial": start.isoformat(),
            "tempo_final": end.isoformat(),
        })

    rows.sort(key=lambda r: r["estacao"])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "estacao","lon_obs","lat_obs","lon_mod","lat_mod","tempo_inicial","tempo_final"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Total estacoes com interseccao: {len(rows)}")
    print(f"Arquivo salvo: {OUTPUT_CSV}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # permitir passar diretorios na linha de comando
        ROOT_DIRS = sys.argv[1:]
    main()