#!/usr/bin/env python3
"""
Script para criar mapas de validação das estatísticas P99 (eventos extremos)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os

def create_p99_validation_maps():
    """
    Cria mapas de validação para estatísticas P99 (eventos extremos)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    
    if not os.path.exists(stats_file):
        print(f"Erro: Arquivo {stats_file} não encontrado!")
        return
    
    # Ler dados
    df = pd.read_csv(stats_file)
    print(f"📊 Dados carregados: {len(df)} estações")
    
    # Calcular métricas P99
    df['nbias_p99_pct'] = (df['bias_p99'] / df['p99_obs']) * 100
    df['nrmse_p99_pct'] = (df['rmse_p99'] / df['p99_obs']) * 100
    df['si_p99'] = df['rmse_p99'] / df['p99_obs']
    
    # Classificar por tipo de estação
    df['tipo_estacao'] = df['estacao'].apply(lambda x: 'NDBC' if str(x).startswith(('4', '5')) else 'EMODNET')
    
    # Obter coordenadas das estações (usando dados do validation_metrics.csv se disponível)
    metrics_file = os.path.join(script_dir, 'validation_metrics.csv')
    if os.path.exists(metrics_file):
        df_coords = pd.read_csv(metrics_file)
        df = df.merge(df_coords[['station', 'longitude', 'latitude']], 
                      left_on='estacao', right_on='station', how='left')
    else:
        print("⚠️  Arquivo validation_metrics.csv não encontrado. Usando coordenadas padrão.")
        # Coordenadas aproximadas baseadas nos IDs das estações
        df['longitude'] = df['estacao'].apply(get_approximate_longitude)
        df['latitude'] = df['estacao'].apply(get_approximate_latitude)
    
    # Filtrar estações com coordenadas válidas
    df = df.dropna(subset=['longitude', 'latitude'])
    print(f"📍 Estações com coordenadas: {len(df)}")
    
    # Calcular estatísticas globais P99
    nbias_p99_mean = df['nbias_p99_pct'].mean()
    nrmse_p99_mean = df['nrmse_p99_pct'].mean()
    corr_p99_mean = df['corr_p99'].mean()
    
    print(f"\n📊 ESTATÍSTICAS GLOBAIS P99:")
    print(f"   NBIAS P99 médio: {nbias_p99_mean:.2f}%")
    print(f"   NRMSE P99 médio: {nrmse_p99_mean:.2f}%")
    print(f"   Correlação P99 média: {corr_p99_mean:.4f}")
    
    # Criar os mapas
    metrics = [
        ('nbias_p99_pct', 'NBIAS P99 (%)', 'RdBu_r', (-40, 40)),
        ('nrmse_p99_pct', 'NRMSE P99 (%)', 'YlOrRd', (0, 60)),
        ('corr_p99', 'Correlação P99', 'RdYlBu', (-0.5, 1.0))
    ]
    fig = plt.figure(figsize=(18, 14))
    # 1. NBIAS P99 Map
    ax1 = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    ax1.set_global()
    ax1.coastlines(resolution='110m', linewidth=0.8, color='black')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax1.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax1.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax1.set_title('Normalized Bias P99 (NBIAS P99)', fontsize=14, fontweight='bold', pad=10)
    scatter1 = ax1.scatter(df['longitude'], df['latitude'], c=df['nbias_p99_pct'], cmap='RdBu_r', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=-40, vmax=40, zorder=5, transform=ccrs.PlateCarree())
    cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='vertical', pad=0.05, shrink=0.7)
    cbar1.set_label('NBIAS P99 (%)', fontsize=11, fontweight='bold')
    ax1.text(0.02, 0.95, f'NBIAS P99 = {nbias_p99_mean:.2f}%', transform=ax1.transAxes, fontsize=13, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.legend(['NDBC', 'EMODNET'], fontsize=10, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    # 2. NRMSE P99 Map
    ax2 = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    ax2.set_global()
    ax2.coastlines(resolution='110m', linewidth=0.8, color='black')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax2.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax2.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax2.set_title('Normalized Root Mean Square Error P99 (NRMSE P99)', fontsize=14, fontweight='bold', pad=10)
    scatter2 = ax2.scatter(df['longitude'], df['latitude'], c=df['nrmse_p99_pct'], cmap='YlOrRd', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=0, vmax=60, zorder=5, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='vertical', pad=0.05, shrink=0.7)
    cbar2.set_label('NRMSE P99 (%)', fontsize=11, fontweight='bold')
    ax2.text(0.02, 0.95, f'NRMSE P99 = {nrmse_p99_mean:.2f}%', transform=ax2.transAxes, fontsize=13, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 3. Correlation P99 Map
    ax3 = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
    ax3.set_global()
    ax3.coastlines(resolution='110m', linewidth=0.8, color='black')
    ax3.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax3.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
    ax3.gridlines(draw_labels=False, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    ax3.set_title('Correlation Coefficient P99', fontsize=14, fontweight='bold', pad=10)
    scatter3 = ax3.scatter(df['longitude'], df['latitude'], c=df['corr_p99'], cmap='RdYlBu', s=150, alpha=0.8, edgecolors='black', linewidth=0.8, vmin=-0.5, vmax=1.0, zorder=5, transform=ccrs.PlateCarree())
    cbar3 = plt.colorbar(scatter3, ax=ax3, orientation='vertical', pad=0.05, shrink=0.7)
    cbar3.set_label('Correlation P99', fontsize=11, fontweight='bold')
    ax3.text(0.02, 0.95, f'Correlation P99 = {corr_p99_mean:.4f}', transform=ax3.transAxes, fontsize=13, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('WW3 Model Validation - P99 Statistics (Extreme Events)', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    output_file = os.path.join(script_dir, 'validation_maps_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Mapa P99 salvo: {output_file}")

    # Criar mapa adicional com identificação de estações problemáticas P99
    create_problematic_p99_map(df, script_dir)

    plt.show()

def create_problematic_p99_map(df, script_dir):
    """
    Cria mapa destacando estações com problemas no P99
    """
    print("\n🔍 Criando mapa de estações problemáticas P99...")
    
    # Identificar estações problemáticas
    df['p99_problematic'] = (
        (df['corr_p99'] < 0) |  # Correlação negativa
        (df['nrmse_p99_pct'] > 40) |  # NRMSE P99 muito alto
        (abs(df['nbias_p99_pct']) > 30)  # NBIAS P99 muito alto
    )
    
    problematic = df[df['p99_problematic']]
    normal = df[~df['p99_problematic']]
    
    print(f"🔴 Estações problemáticas P99: {len(problematic)}")
    print(f"✅ Estações normais P99: {len(normal)}")
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10), 
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Configurar mapa
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='white', alpha=0.8)
    
    # Grade
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                     linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
    
    # Plotar estações normais
    if len(normal) > 0:
        ax.scatter(normal['longitude'], normal['latitude'], 
                  c='lightblue', s=30, alpha=0.6, 
                  edgecolors='gray', linewidth=0.5,
                  label=f'Normal P99 ({len(normal)})', transform=ccrs.PlateCarree())
    
    # Plotar estações problemáticas
    if len(problematic) > 0:
        # Separar por tipo de problema
        neg_corr = problematic[problematic['corr_p99'] < 0]
        high_nrmse = problematic[problematic['nrmse_p99_pct'] > 40]
        high_bias = problematic[abs(problematic['nbias_p99_pct']) > 30]
        
        # Plotar correlação negativa
        if len(neg_corr) > 0:
            ax.scatter(neg_corr['longitude'], neg_corr['latitude'], 
                      c='red', s=80, marker='x', linewidth=2,
                      label=f'Corr P99 < 0 ({len(neg_corr)})', transform=ccrs.PlateCarree())
        
        # Plotar NRMSE alto
        if len(high_nrmse) > 0:
            ax.scatter(high_nrmse['longitude'], high_nrmse['latitude'], 
                      c='orange', s=60, marker='^', 
                      edgecolors='black', linewidth=0.5,
                      label=f'NRMSE P99 > 40% ({len(high_nrmse)})', transform=ccrs.PlateCarree())
        
        # Plotar BIAS alto
        if len(high_bias) > 0:
            ax.scatter(high_bias['longitude'], high_bias['latitude'], 
                      c='purple', s=60, marker='s', 
                      edgecolors='black', linewidth=0.5,
                      label=f'|NBIAS P99| > 30% ({len(high_bias)})', transform=ccrs.PlateCarree())
    
    ax.set_title('WW3 Model Validation - Problematic P99 Stations (Extreme Events)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Salvar
    output_file = os.path.join(script_dir, 'validation_map_problematic_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Mapa problemáticas P99 salvo: {output_file}")

def get_approximate_longitude(station_id):
    """Estimar longitude baseada no ID da estação"""
    station_str = str(station_id)
    if station_str.startswith('41'):
        return -70  # Atlântico Norte Oeste
    elif station_str.startswith('42'):
        return -75  # Golfo do México
    elif station_str.startswith('44'):
        return -70  # Costa Leste EUA
    elif station_str.startswith('46'):
        return -125  # Costa Oeste EUA
    elif station_str.startswith('51'):
        return -155  # Havaí
    elif station_str.startswith('52'):
        return -135  # Alasca
    elif station_str.startswith('55'):
        return 145   # Austrália
    elif station_str.startswith('56'):
        return 115   # Oceano Índico
    elif station_str.startswith('61'):
        return -45   # Atlântico Sul
    elif station_str.startswith('62'):
        return 5     # Atlântico/Mediterrâneo
    elif station_str.startswith('63'):
        return 10    # Mediterrâneo
    elif station_str.startswith('66'):
        return -20   # Atlântico Tropical
    else:
        return 0

def get_approximate_latitude(station_id):
    """Estimar latitude baseada no ID da estação"""
    station_str = str(station_id)
    if station_str.startswith('41'):
        return 35   # Atlântico Norte
    elif station_str.startswith('42'):
        return 25   # Golfo do México
    elif station_str.startswith('44'):
        return 40   # Costa Leste EUA
    elif station_str.startswith('46'):
        return 40   # Costa Oeste EUA
    elif station_str.startswith('51'):
        return 20   # Havaí
    elif station_str.startswith('52'):
        return 55   # Alasca
    elif station_str.startswith('55'):
        return -35  # Austrália
    elif station_str.startswith('56'):
        return -30  # Oceano Índico
    elif station_str.startswith('61'):
        return -25  # Atlântico Sul
    elif station_str.startswith('62'):
        return 50   # Atlântico/Mediterrâneo
    elif station_str.startswith('63'):
        return 40   # Mediterrâneo
    elif station_str.startswith('66'):
        return 15   # Atlântico Tropical
    elif station_str.startswith('130'):
        return -25  # Brasil
    else:
        return 0

if __name__ == "__main__":
    create_p99_validation_maps()