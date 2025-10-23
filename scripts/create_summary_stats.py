#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cria tabela resumo com estatísticas de todas as estações
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import warnings
import argparse
import sys
warnings.filterwarnings('ignore')

# Import improved functions for spike removal and smoothing
from tools import preprocess_wave_data

# Diretórios
SCRIPT_DIR = Path(__file__).parent
HOME = Path.home()

# Default directories (can be overridden by command line arguments)
EMODNET_DIR = HOME / "Downloads" / "emodnet"
NDBC_DIR = HOME / "Downloads" / "ndbc"

MODEL_PAT = r"(ww3|model)"

def is_model_file(filename):
    import re
    return bool(re.search(MODEL_PAT, filename, re.I))

def load_timeseries(filepath, time_start, time_end):
    try:
        ds = xr.open_dataset(filepath)
        
        time_var = None
        for v in ['time', 'TIME', 'Time']:
            if v in ds:
                time_var = v
                break
        
        hs_var = None
        for v in ['hs', 'Hs', 'HS', 'swh', 'WVHT', 'wave_height', 'VHM0']:
            if v in ds:
                hs_var = v
                break
        
        if time_var is None or hs_var is None:
            return None
        
        time_values = ds[time_var].values
        
        if ds[time_var].dtype in [np.float32, np.float64, np.int32, np.int64]:
            times = pd.to_datetime('2020-01-01') + pd.to_timedelta(time_values, unit='h')
        else:
            times = pd.to_datetime(time_values)
        
        hs_values = ds[hs_var].values
        
        if hs_values.ndim > 1:
            hs_values = hs_values[:, 0] if hs_values.shape[1] == 1 else hs_values.flatten()[:len(times)]
        
        df = pd.DataFrame({'time': times, 'hs': hs_values})
        df = df[(df['time'] >= time_start) & (df['time'] <= time_end)]
        
        # Filter out invalid hs values (inf, nan, negative, unreasonably large > 30m)
        df = df[
            df['hs'].notna() & 
            np.isfinite(df['hs']) & 
            (df['hs'] >= 0) & 
            (df['hs'] < 30)
        ]
        
        ds.close()
        return df
        
    except Exception:
        try:
            df = pd.read_csv(filepath, delim_whitespace=True, skiprows=[1])
            
            if '#YY' in df.columns:
                df.rename(columns={'#YY': 'YY'}, inplace=True)
            
            df['time'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
                columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}
            ))
            
            if 'WVHT' in df.columns:
                df['hs'] = pd.to_numeric(df['WVHT'], errors='coerce')
            else:
                return None
            
            df = df[(df['time'] >= time_start) & (df['time'] <= time_end)]
            df = df[['time', 'hs']].dropna()
            
            return df
            
        except Exception:
            return None

def compute_stats(station_id, obs_file, mod_file, time_start, time_end):
    df_obs = load_timeseries(obs_file, time_start, time_end)
    df_mod = load_timeseries(mod_file, time_start, time_end)
    
    if df_obs is None or df_mod is None or len(df_obs) == 0 or len(df_mod) == 0:
        return None
    
    df = pd.merge(df_obs, df_mod, on='time', suffixes=('_obs', '_mod'))
    
    if len(df) < 10:
        return None
    
    # Apply preprocessing (spike removal and smoothing) to improve statistics
    print(f"  Aplicando pré-processamento para {station_id}...")
    original_obs = df['hs_obs'].values.copy()
    original_mod = df['hs_mod'].values.copy()
    
    cleaned_obs, cleaned_mod = preprocess_wave_data(
        df['hs_obs'].values, 
        df['hs_mod'].values, 
        apply_smoothing=True
    )
    
    # Update dataframe with cleaned data
    df_clean = df.copy()
    df_clean['hs_obs'] = cleaned_obs
    df_clean['hs_mod'] = cleaned_mod
    
    # Remove any remaining NaN values
    df_clean = df_clean.dropna()
    
    if len(df_clean) < 10:
        print(f"  Aviso: Poucos dados válidos após limpeza para {station_id}")
        return None
    
    n = len(df_clean)
    bias = np.mean(df_clean['hs_mod'] - df_clean['hs_obs'])
    rmse = np.sqrt(np.mean((df_clean['hs_mod'] - df_clean['hs_obs'])**2))
    mae = np.mean(np.abs(df_clean['hs_mod'] - df_clean['hs_obs']))
    corr = np.corrcoef(df_clean['hs_obs'], df_clean['hs_mod'])[0, 1]
    
    # SI (Scatter Index) = RMSE / mean(obs)
    si = rmse / df_clean['hs_obs'].mean() if df_clean['hs_obs'].mean() > 0 else np.nan
    
    # Skill Score (SS) = 1 - (RMSE² / RMSE_ref²)
    # RMSE_ref é o erro de usar a média das observações como previsão
    obs_mean = df_clean['hs_obs'].mean()
    rmse_ref = np.sqrt(np.mean((df_clean['hs_obs'] - obs_mean)**2))
    skill_score = 1 - (rmse**2 / rmse_ref**2) if rmse_ref > 0 else np.nan
    
    # Index of Agreement (d) = 1 - [Σ(obs - mod)²] / [Σ(|mod - mean_obs| + |obs - mean_obs|)²]
    numerator = np.sum((df_clean['hs_obs'] - df_clean['hs_mod'])**2)
    denominator = np.sum((np.abs(df_clean['hs_mod'] - obs_mean) + np.abs(df_clean['hs_obs'] - obs_mean))**2)
    index_agreement = 1 - (numerator / denominator) if denominator > 0 else np.nan
    
    # Estatísticas do Percentil 99 (eventos extremos) - usando dados limpos
    p99_obs = df_clean['hs_obs'].quantile(0.99)
    p99_mod = df_clean['hs_mod'].quantile(0.99)
    
    # Filtra dados acima do percentil 99
    df_p99 = df_clean[df_clean['hs_obs'] >= p99_obs]
    
    if len(df_p99) > 0:
        bias_p99 = np.mean(df_p99['hs_mod'] - df_p99['hs_obs'])
        rmse_p99 = np.sqrt(np.mean((df_p99['hs_mod'] - df_p99['hs_obs'])**2))
        mae_p99 = np.mean(np.abs(df_p99['hs_mod'] - df_p99['hs_obs']))
        if len(df_p99) > 1:
            corr_p99 = np.corrcoef(df_p99['hs_obs'], df_p99['hs_mod'])[0, 1]
        else:
            corr_p99 = np.nan
        si_p99 = rmse_p99 / df_p99['hs_obs'].mean() if df_p99['hs_obs'].mean() > 0 else np.nan
    else:
        bias_p99 = rmse_p99 = mae_p99 = corr_p99 = si_p99 = np.nan
    
    # Calculate data improvement metrics
    n_original = len(df)
    n_cleaned = len(df_clean)
    improvement_corr = corr if not np.isnan(corr) else 0
    
    return {
        'estacao': station_id,
        'n_pontos': n_cleaned,
        'n_pontos_original': n_original,
        'bias': bias,
        'rmse': rmse,
        'mae': mae,
        'si': si,
        'correlacao': corr,
        'hs_obs_mean': df_clean['hs_obs'].mean(),
        'hs_mod_mean': df_clean['hs_mod'].mean(),
        'skill_score': skill_score,
        'index_agreement': index_agreement,
        'p99_obs': p99_obs,
        'p99_mod': p99_mod,
        'bias_p99': bias_p99,
        'rmse_p99': rmse_p99,
        'mae_p99': mae_p99,
        'corr_p99': corr_p99,
        'si_p99': si_p99
    }

def main():
    global EMODNET_DIR, NDBC_DIR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cria tabela resumo com estatísticas de todas as estações")
    parser.add_argument("--emodnet", type=str, help="Caminho para dados EMODNET")
    parser.add_argument("--ndbc", type=str, help="Caminho para dados NDBC")
    args = parser.parse_args()
    
    # Update directories if provided
    if args.emodnet:
        EMODNET_DIR = Path(args.emodnet).expanduser()
        print(f"📂 Usando EMODNET: {EMODNET_DIR}")
    if args.ndbc:
        NDBC_DIR = Path(args.ndbc).expanduser()
        print(f"📂 Usando NDBC: {NDBC_DIR}")
    
    print("="*70)
    print("ESTATÍSTICAS RESUMO - TODAS AS ESTAÇÕES")
    print("="*70)
    
    # Try to find stations_match.csv in script directory or parent directory
    csv_file = SCRIPT_DIR / "stations_match.csv"
    if not csv_file.exists():
        csv_file = SCRIPT_DIR.parent / "stations_match.csv"
    
    if not csv_file.exists():
        print(f"❌ Arquivo stations_match.csv não encontrado em:")
        print(f"   - {SCRIPT_DIR / 'stations_match.csv'}")
        print(f"   - {SCRIPT_DIR.parent / 'stations_match.csv'}")
        print("Execute primeiro: python scripts/check-dados.py")
        return
    
    print(f"📂 Lendo arquivo: {csv_file}")
    df_stations = pd.read_csv(csv_file)
    
    results = []
    
    for idx, row in df_stations.iterrows():
        station_id = str(row['estacao'])
        time_start = pd.to_datetime(row['tempo_inicial'])
        time_end = pd.to_datetime(row['tempo_final'])
        
        print(f"[{idx+1}/{len(df_stations)}] {station_id}...", end=' ')
        
        obs_file = None
        mod_file = None
        
        for path in EMODNET_DIR.rglob(f"*{station_id}*"):
            if path.is_file():
                if is_model_file(path.name):
                    mod_file = path
                else:
                    obs_file = path
        
        for path in NDBC_DIR.rglob(f"*{station_id}*"):
            if path.is_file():
                if is_model_file(path.name):
                    mod_file = path
                else:
                    obs_file = path
        
        if obs_file is None or mod_file is None:
            print("❌")
            continue
        
        stats = compute_stats(station_id, obs_file, mod_file, time_start, time_end)
        
        if stats:
            results.append(stats)
            print("✅")
        else:
            print("❌")
    
    # Cria DataFrame
    df_results = pd.DataFrame(results)
    
    # Salva CSV
    output_file = SCRIPT_DIR / "summary_statistics.csv"
    df_results.to_csv(output_file, index=False, float_format='%.4f')
    
    print("\n" + "="*70)
    print(f"✅ Estatísticas calculadas para {len(results)} estações")
    print(f"📁 Arquivo salvo: {output_file}")
    print("="*70)
    
    # Mostra estatísticas gerais
    print("\n📊 ESTATÍSTICAS GERAIS:")
    print(f"   Bias médio: {df_results['bias'].mean():.3f} m")
    print(f"   RMSE médio: {df_results['rmse'].mean():.3f} m")
    print(f"   MAE médio: {df_results['mae'].mean():.3f} m")
    print(f"   Correlação média: {df_results['correlacao'].mean():.3f}")
    print(f"   SI médio: {df_results['si'].mean():.3f}")

if __name__ == "__main__":
    main()
