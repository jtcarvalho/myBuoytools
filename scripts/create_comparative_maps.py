#!/usr/bin/env python3
"""
Script para criar mapas comparativos: Estatísticas Gerais vs P99
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os

def create_comparative_maps():
    """
    Cria mapas comparativos entre estatísticas gerais e P99
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    metrics_file = os.path.join(script_dir, 'validation_metrics.csv')
    
    if not os.path.exists(stats_file):
        print(f"Erro: Arquivo {stats_file} não encontrado!")
        return
    
    # Ler dados
    df = pd.read_csv(stats_file)
    df_coords = pd.read_csv(metrics_file)
    
    # Merge coordenadas
    df = df.merge(df_coords[['station', 'longitude', 'latitude']], 
                  left_on='estacao', right_on='station', how='left')
    
    # Calcular métricas
    df['nrmse_pct'] = df['si'] * 100
    df['nbias_pct'] = (df['bias'] / df['hs_obs_mean']) * 100
    df['nbias_p99_pct'] = (df['bias_p99'] / df['p99_obs']) * 100
    df['nrmse_p99_pct'] = (df['rmse_p99'] / df['p99_obs']) * 100
    
    # Calcular diferenças
    df['diff_nrmse'] = df['nrmse_p99_pct'] - df['nrmse_pct']
    df['diff_corr'] = df['correlacao'] - df['corr_p99']
    df['diff_nbias'] = df['nbias_p99_pct'] - df['nbias_pct']
    
    # Filtrar dados válidos
    df = df.dropna(subset=['longitude', 'latitude'])
    
    print(f"📊 Criando mapas comparativos para {len(df)} estações")
    print(f"📈 Diferença NRMSE média (P99 - Geral): {df['diff_nrmse'].mean():.2f}%")
    print(f"📈 Diferença Correlação média (Geral - P99): {df['diff_corr'].mean():.4f}")
    
    # Criar figura com 4 subplots (2x2) - removendo correlação
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    fig.suptitle('WW3 Model Validation: General vs P99 Statistics Comparison', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Definir métricas para comparação (sem correlação)
    comparisons = [
        # (métrica_geral, métrica_p99, título_geral, título_p99, colormap, vrange, label)
        ('nrmse_pct', 'nrmse_p99_pct', 'NRMSE General (%)', 'NRMSE P99 (%)', 'YlOrRd', (0, 60), ['(a)', '(b)']),
        ('nbias_pct', 'nbias_p99_pct', 'NBIAS General (%)', 'NBIAS P99 (%)', 'RdBu_r', (-40, 40), ['(c)', '(d)'])
    ]
    
    for row, (metric_gen, metric_p99, title_gen, title_p99, colormap, vrange, labels) in enumerate(comparisons):
        # Mapa estatísticas gerais (coluna 0)
        ax_gen = axes[row, 0]
        create_single_map(ax_gen, df, metric_gen, title_gen, colormap, vrange, labels[0])
        
        # Mapa estatísticas P99 (coluna 1)
        ax_p99 = axes[row, 1]
        create_single_map(ax_p99, df, metric_p99, title_p99, colormap, vrange, labels[1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Deixar espaço para o título
    
    # Salvar
    output_file = os.path.join(script_dir, 'validation_maps_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Mapa comparativo salvo: {output_file}")
    
    # Criar mapa de diferenças
    create_difference_maps(df, script_dir)
    
    plt.show()

def create_single_map(ax, df, metric, title, colormap, vrange, panel_label):
    """
    Cria um único mapa com identificação do painel
    """
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
    
    # Plotar dados
    scatter = ax.scatter(df['longitude'], df['latitude'], 
                        c=df[metric], s=50, 
                        cmap=colormap, vmin=vrange[0], vmax=vrange[1],
                        edgecolors='black', linewidth=0.5, alpha=0.8,
                        transform=ccrs.PlateCarree())
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label(title, fontsize=10, fontweight='bold')
    
    # Estatística média
    mean_val = df[metric].mean()
    if 'NRMSE' in title or 'NBIAS' in title:
        mean_text = f"{title.split()[0]} = {mean_val:.2f}%"
    else:
        mean_text = f"{title.split()[0]} = {mean_val:.4f}"
    
    ax.text(0.02, 0.98, mean_text, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
           fontsize=12, fontweight='bold', verticalalignment='top')
    
    # Adicionar identificação do painel
    ax.text(0.02, 0.02, panel_label, transform=ax.transAxes,
           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8),
           fontsize=14, fontweight='bold', color='white', 
           verticalalignment='bottom', horizontalalignment='left')

def create_difference_maps(df, script_dir):
    """
    Cria mapas das diferenças entre estatísticas gerais e P99
    """
    print("\n🔄 Criando mapas de diferenças...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), 
                            subplot_kw={'projection': ccrs.PlateCarree()})
    
    fig.suptitle('WW3 Model Validation: Differences (P99 - General Statistics)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    differences = [
        ('diff_nrmse', 'ΔNRMSE (P99 - General) (%)', 'RdYlBu_r', (-20, 20), '(e)'),
        ('diff_nbias', 'ΔNBIAS (P99 - General) (%)', 'RdBu_r', (-40, 40), '(f)')
    ]
    
    for i, (metric, title, colormap, vrange, panel_label) in enumerate(differences):
        ax = axes[i]
        
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
        
        # Plotar diferenças
        scatter = ax.scatter(df['longitude'], df['latitude'], 
                            c=df[metric], s=60, 
                            cmap=colormap, vmin=vrange[0], vmax=vrange[1],
                            edgecolors='black', linewidth=0.5, alpha=0.8,
                            transform=ccrs.PlateCarree())
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                           pad=0.05, shrink=0.8, aspect=30)
        cbar.set_label(title, fontsize=10, fontweight='bold')
        
        # Estatística média
        mean_val = df[metric].mean()
        if 'NRMSE' in title or 'NBIAS' in title:
            mean_text = f"Δ Mean = {mean_val:.2f}%"
        else:
            mean_text = f"Δ Mean = {mean_val:.4f}"
        
        ax.text(0.02, 0.98, mean_text, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
               fontsize=12, fontweight='bold', verticalalignment='top')
        
        # Adicionar identificação do painel
        ax.text(0.02, 0.02, panel_label, transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8),
               fontsize=14, fontweight='bold', color='white', 
               verticalalignment='bottom', horizontalalignment='left')
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Deixar espaço para o título
    
    # Salvar
    output_file = os.path.join(script_dir, 'validation_maps_differences.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Mapa de diferenças salvo: {output_file}")

if __name__ == "__main__":
    create_comparative_maps()