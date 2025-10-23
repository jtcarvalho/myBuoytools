#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scatter plots de todas as estações - Observações vs Modelo
Compara Hs (altura significativa de onda) entre dados observados e modelados
"""

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Diretórios
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "scatter_plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Diretórios de dados
HOME = Path.home()
EMODNET_DIR = HOME / "Downloads" / "emodnet"
NDBC_DIR = HOME / "Downloads" / "ndbc"

# Padrão para detectar arquivos de modelo
MODEL_PAT = r"(ww3|model)"

def is_model_file(filename):
    """Verifica se o arquivo é um arquivo de modelo"""
    import re
    return bool(re.search(MODEL_PAT, filename, re.I))

def guess_station_id(filepath):
    """Extrai o ID da estação do nome do arquivo"""
    import re
    fname = Path(filepath).stem
    
    # Se for arquivo de modelo, remove o prefixo
    if is_model_file(fname):
        parts = re.split(r'[_\-\.]', fname)
        for i, part in enumerate(parts):
            if re.search(MODEL_PAT, part, re.I) and i + 1 < len(parts):
                return parts[i + 1]
    
    # Tenta extrair números
    match = re.search(r'\d+', fname)
    if match:
        return match.group()
    
    return fname

def load_timeseries(filepath, time_start, time_end):
    """
    Carrega série temporal de Hs do arquivo
    Retorna DataFrame com timestamp e hs
    """
    try:
        # Tenta carregar como NetCDF
        ds = xr.open_dataset(filepath)
        
        # Identifica variáveis
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
        
        # Lê os dados
        time_values = ds[time_var].values
        
        # Converte tempo se necessário (EMODNET)
        if ds[time_var].dtype in [np.float32, np.float64, np.int32, np.int64]:
            times = pd.to_datetime('2020-01-01') + pd.to_timedelta(time_values, unit='h')
        else:
            times = pd.to_datetime(time_values)
        
        hs_values = ds[hs_var].values
        
        # Se multidimensional, pega a primeira dimensão espacial
        if hs_values.ndim > 1:
            hs_values = hs_values[:, 0] if hs_values.shape[1] == 1 else hs_values.flatten()[:len(times)]
        
        # Cria DataFrame
        df = pd.DataFrame({
            'time': times,
            'hs': hs_values
        })
        
        # Filtra período de interesse
        df = df[(df['time'] >= time_start) & (df['time'] <= time_end)]
        
        # Remove NaNs
        df = df.dropna()
        
        ds.close()
        return df
        
    except Exception as e:
        # Tenta como arquivo de texto
        try:
            # Lê arquivo de texto (formato NDBC)
            df = pd.read_csv(filepath, delim_whitespace=True, skiprows=[1])
            
            # Cria timestamp
            if '#YY' in df.columns:
                df.rename(columns={'#YY': 'YY'}, inplace=True)
            
            df['time'] = pd.to_datetime(df[['YY', 'MM', 'DD', 'hh', 'mm']].rename(
                columns={'YY': 'year', 'MM': 'month', 'DD': 'day', 'hh': 'hour', 'mm': 'minute'}
            ))
            
            # Pega Hs
            if 'WVHT' in df.columns:
                df['hs'] = pd.to_numeric(df['WVHT'], errors='coerce')
            else:
                return None
            
            # Filtra período e remove NaNs
            df = df[(df['time'] >= time_start) & (df['time'] <= time_end)]
            df = df[['time', 'hs']].dropna()
            
            return df
            
        except Exception:
            return None

def make_scatter_plot(station_id, obs_file, mod_file, time_start, time_end, output_file):
    """
    Cria scatter plot comparando observações vs modelo
    """
    # Carrega dados
    df_obs = load_timeseries(obs_file, time_start, time_end)
    df_mod = load_timeseries(mod_file, time_start, time_end)
    
    if df_obs is None or df_mod is None:
        print(f"  ❌ Erro ao carregar dados para estação {station_id}")
        return False
    
    if len(df_obs) == 0 or len(df_mod) == 0:
        print(f"  ❌ Sem dados para estação {station_id}")
        return False
    
    # Merge nos timestamps comuns
    df = pd.merge(df_obs, df_mod, on='time', suffixes=('_obs', '_mod'))
    
    if len(df) < 10:
        print(f"  ❌ Poucos pontos ({len(df)}) para estação {station_id}")
        return False
    
    # Estatísticas
    n = len(df)
    bias = np.mean(df['hs_mod'] - df['hs_obs'])
    rmse = np.sqrt(np.mean((df['hs_mod'] - df['hs_obs'])**2))
    corr = np.corrcoef(df['hs_obs'], df['hs_mod'])[0, 1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(df['hs_obs'], df['hs_mod'], alpha=0.5, s=20, edgecolors='none')
    
    # Linha 1:1
    max_val = max(df['hs_obs'].max(), df['hs_mod'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', lw=2, label='1:1 line')
    
    # Configurações
    ax.set_xlabel('Hs Observado (m)', fontsize=12)
    ax.set_ylabel('Hs Modelo (m)', fontsize=12)
    ax.set_title(f'Estação {station_id}\n{time_start.date()} a {time_end.date()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Estatísticas no gráfico
    textstr = f'N = {n}\nBias = {bias:.3f} m\nRMSE = {rmse:.3f} m\nCorr = {corr:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ {station_id}: N={n}, Bias={bias:.3f}m, RMSE={rmse:.3f}m, R={corr:.3f}")
    return True

def main():
    """Função principal"""
    print("="*70)
    print("SCATTER PLOTS - OBSERVAÇÕES VS MODELO")
    print("="*70)
    
    # Lê arquivo com estações
    csv_file = SCRIPT_DIR / "stations_match.csv"
    df_stations = pd.read_csv(csv_file)
    
    print(f"\n📊 Total de estações a processar: {len(df_stations)}\n")
    
    # Contador de sucesso
    success_count = 0
    fail_count = 0
    
    # Processa cada estação
    for idx, row in df_stations.iterrows():
        station_id = str(row['estacao'])
        time_start = pd.to_datetime(row['tempo_inicial'])
        time_end = pd.to_datetime(row['tempo_final'])
        
        print(f"[{idx+1}/{len(df_stations)}] Processando estação {station_id}...")
        
        # Busca arquivos
        obs_file = None
        mod_file = None
        
        # Busca em EMODNET
        for path in EMODNET_DIR.rglob(f"*{station_id}*"):
            if path.is_file():
                if is_model_file(path.name):
                    mod_file = path
                else:
                    obs_file = path
        
        # Busca em NDBC
        for path in NDBC_DIR.rglob(f"*{station_id}*"):
            if path.is_file():
                if is_model_file(path.name):
                    mod_file = path
                else:
                    obs_file = path
        
        if obs_file is None or mod_file is None:
            print(f"  ❌ Arquivos não encontrados")
            fail_count += 1
            continue
        
        # Cria scatter plot
        output_file = OUTPUT_DIR / f"scatter_{station_id}.png"
        success = make_scatter_plot(station_id, obs_file, mod_file, 
                                     time_start, time_end, output_file)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print("\n" + "="*70)
    print(f"✅ Sucesso: {success_count} estações")
    print(f"❌ Falhas: {fail_count} estações")
    print(f"📁 Gráficos salvos em: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
