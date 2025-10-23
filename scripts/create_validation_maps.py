#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mapas globais com métricas de validação por estação
Similar ao formato exp_01-st4-uost-psi-400s-era5-b143-ic5-ref1
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
STATIONS_FILE = SCRIPT_DIR.parent / "stations_match.csv"
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_FILE = OUTPUT_DIR / "validation_maps.png"

def calculate_nrmse_nbias(df):
    """Calcula NRMSE e NBIAS normalizados em percentual"""
    df['nrmse'] = (df['rmse'] / df['hs_obs_mean']) * 100
    df['nbias'] = (df['bias'] / df['hs_obs_mean']) * 100
    return df

def create_global_map(ax, title, metric_label):
    """Cria base do mapa global"""
    ax.set_global()
    ax.coastlines(resolution='110m', linewidth=0.8, color='black')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax.gridlines(draw_labels=False, linewidth=0.5, color='gray', 
                 alpha=0.5, linestyle='--')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    return ax

def main():
    print("="*70)
    print("CRIANDO MAPAS GLOBAIS DE VALIDAÇÃO")
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
    
    print(f"\n📊 Estações válidas para mapas: {len(df)}")
    print(f"   NRMSE médio: {df['nrmse'].mean():.2f}%")
    print(f"   NBIAS médio: {df['nbias'].mean():.2f}%")
    print(f"   Correlação média: {df['correlacao'].mean():.3f}")
    
    # Cria figura com 3 subplots
    fig = plt.figure(figsize=(18, 14))
    
    # =====================================================================
    # 1. NRMSE Map
    # =====================================================================
    ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    create_global_map(ax1, 'Normalized Root Mean Square Error (NRMSE)', 'NRMSE (%)')
    
    # Plot pontos
    scatter1 = ax1.scatter(df['lon_obs'], df['lat_obs'], 
                          c=df['nrmse'], cmap='YlOrRd', 
                          s=150, alpha=0.8, edgecolors='black', linewidth=0.8,
                          vmin=0, vmax=60, zorder=5,
                          transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='vertical', 
                        pad=0.05, shrink=0.7)
    cbar1.set_label('NRMSE (%)', fontsize=11, fontweight='bold')
    
    # Texto com média global
    nrmse_mean = df['nrmse'].mean()
    ax1.text(0.02, 0.95, f'NRMSE = {nrmse_mean:.2f}%', 
            transform=ax1.transAxes, fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =====================================================================
    # 2. NBIAS Map
    # =====================================================================
    ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    create_global_map(ax2, 'Normalized Bias (NBIAS)', 'NBIAS (%)')
    
    # Plot pontos com colormap divergente
    scatter2 = ax2.scatter(df['lon_obs'], df['lat_obs'], 
                          c=df['nbias'], cmap='RdBu_r', 
                          s=150, alpha=0.8, edgecolors='black', linewidth=0.8,
                          vmin=-40, vmax=40, zorder=5,
                          transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='vertical', 
                        pad=0.05, shrink=0.7)
    cbar2.set_label('NBIAS (%)', fontsize=11, fontweight='bold')
    
    # Texto com média global
    nbias_mean = df['nbias'].mean()
    ax2.text(0.02, 0.95, f'NBIAS = {nbias_mean:.2f}%', 
            transform=ax2.transAxes, fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =====================================================================
    # 3. Correlation Map
    # =====================================================================
    ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
    create_global_map(ax3, 'Correlation Coefficient (R)', 'Correlation')
    
    # Plot pontos
    scatter3 = ax3.scatter(df['lon_obs'], df['lat_obs'], 
                          c=df['correlacao'], cmap='RdYlGn', 
                          s=150, alpha=0.8, edgecolors='black', linewidth=0.8,
                          vmin=0.3, vmax=1.0, zorder=5,
                          transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar3 = plt.colorbar(scatter3, ax=ax3, orientation='vertical', 
                        pad=0.05, shrink=0.7)
    cbar3.set_label('Correlation', fontsize=11, fontweight='bold')
    
    # Texto com média global
    corr_mean = df['correlacao'].mean()
    ax3.text(0.02, 0.95, f'R = {corr_mean:.3f}', 
            transform=ax3.transAxes, fontsize=13, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Título geral
    fig.suptitle('WW3 Model Validation - Global Metrics (2020)', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Ajusta layout
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Salva
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✅ Mapa salvo: {OUTPUT_FILE}")
    
    # Estatísticas por região (opcional)
    print("\n" + "="*70)
    print("📊 ESTATÍSTICAS POR REGIÃO:")
    print("="*70)
    
    # Define regiões
    regions = {
        'North Atlantic': ((df['lon_obs'] >= -80) & (df['lon_obs'] <= 0) & 
                          (df['lat_obs'] >= 30) & (df['lat_obs'] <= 70)),
        'North Pacific': ((df['lon_obs'] >= 120) | (df['lon_obs'] <= -100)) & 
                        (df['lat_obs'] >= 20) & (df['lat_obs'] <= 65),
        'Tropical Atlantic': ((df['lon_obs'] >= -60) & (df['lon_obs'] <= 0) & 
                             (df['lat_obs'] >= 0) & (df['lat_obs'] <= 30)),
        'Southern Ocean': (df['lat_obs'] <= -30),
    }
    
    for region_name, mask in regions.items():
        region_data = df[mask]
        if len(region_data) > 0:
            print(f"\n🌍 {region_name} ({len(region_data)} estações):")
            print(f"   NRMSE: {region_data['nrmse'].mean():6.2f}%")
            print(f"   NBIAS: {region_data['nbias'].mean():6.2f}%")
            print(f"   Corr:  {region_data['correlacao'].mean():6.3f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
