#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapas globais individuais - NRMSE, NBIAS e Correlação
Formato similar ao exp_01-st4-uost-psi-400s-era5-b143-ic5-ref1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Diretórios
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "summary_statistics.csv"
STATIONS_FILE = SCRIPT_DIR.parent / "stations_match.csv"  # File is in parent directory
OUTPUT_DIR = SCRIPT_DIR

def calculate_nrmse_nbias(df):
    """Calcula NRMSE e NBIAS normalizados em percentual"""
    df['nrmse'] = (df['rmse'] / df['hs_obs_mean']) * 100
    df['nbias'] = (df['bias'] / df['hs_obs_mean']) * 100
    return df

def create_map(df, metric, cmap, vmin, vmax, title, output_file, label, diverging=False):
    """Cria um mapa individual"""
    
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=0))
    
    # Configuração do mapa
    ax.set_extent([-180, 180, -85, 85], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='110m', linewidth=1.0, color='black', zorder=3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    
    # Grid
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                     alpha=0.5, linestyle='--', zorder=2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 11}
    gl.ylabel_style = {'size': 11}
    
    # Plot pontos
    scatter = ax.scatter(df['lon_obs'], df['lat_obs'], 
                        c=df[metric], cmap=cmap, 
                        s=50, alpha=0.9, edgecolors='black', linewidth=1.2,
                        vmin=vmin, vmax=vmax, zorder=5,
                        transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', 
                       pad=0.04, shrink=0.8, aspect=30)
    cbar.set_label(label, fontsize=13, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    # Texto com métrica média
    metric_mean = df[metric].mean()
    if metric == 'correlacao':
        text = f'{label} = {metric_mean:.2f}'
    else:
        text = f'{label} = {metric_mean:.2f}%'
    
    ax.text(0.98, 0.95, text, 
           transform=ax.transAxes, fontsize=16, fontweight='bold',
           ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor='black', linewidth=1.5))
    
    # Título
    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ {output_file.name}")

def main():
    print("="*70)
    print("CRIANDO MAPAS GLOBAIS INDIVIDUAIS")
    print("="*70)
    
    # Lê dados
    df_stats = pd.read_csv(CSV_FILE)
    df_stations = pd.read_csv(STATIONS_FILE)
    
    # Merge para ter coordenadas
    df = pd.merge(df_stats, df_stations[['estacao', 'lon_obs', 'lat_obs']], 
                  on='estacao', how='left')
    
    # Remove estações com valores inválidos
    df = df[
        ~df['bias'].isin([np.inf, -np.inf]) & 
        ~df['rmse'].isin([np.inf, -np.inf]) &
        ~df['lon_obs'].isin([np.inf, -np.inf]) &
        ~df['lat_obs'].isin([np.inf, -np.inf]) &
        ~df['correlacao'].isna()
    ].copy()
    
    # Calcula NRMSE e NBIAS
    df = calculate_nrmse_nbias(df)
    
    print(f"\n📊 Total de estações: {len(df)}")
    print(f"   NRMSE médio: {df['nrmse'].mean():.2f}%")
    print(f"   NBIAS médio: {df['nbias'].mean():.2f}%")
    print(f"   Correlação média: {df['correlacao'].mean():.3f}\n")
    
    print("Criando mapas...")
    
    # 1. NRMSE Map
    create_map(
        df=df,
        metric='nrmse',
        cmap='YlOrRd',
        vmin=0,
        vmax=60,
        title='WW3 Model Validation - NRMSE (2020)',
        output_file=OUTPUT_DIR / 'validation_map_NRMSE.png',
        label='NRMSE'
    )
    
    # 2. NBIAS Map
    create_map(
        df=df,
        metric='nbias',
        cmap='RdBu_r',
        vmin=-40,
        vmax=40,
        title='WW3 Model Validation - NBIAS (2020)',
        output_file=OUTPUT_DIR / 'validation_map_NBIAS.png',
        label='NBIAS',
        diverging=True
    )
    
    # 3. Correlation Map
    create_map(
        df=df,
        metric='correlacao',
        cmap='RdYlGn',
        vmin=0.3,
        vmax=1.0,
        title='WW3 Model Validation - Correlation (2020)',
        output_file=OUTPUT_DIR / 'validation_map_CORR.png',
        label='R'
    )
    
    print("\n" + "="*70)
    print("✅ TODOS OS MAPAS CRIADOS COM SUCESSO!")
    print("="*70)
    print("\n📁 Arquivos gerados:")
    print("   • validation_map_NRMSE.png")
    print("   • validation_map_NBIAS.png")
    print("   • validation_map_CORR.png")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
