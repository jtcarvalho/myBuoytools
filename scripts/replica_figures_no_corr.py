#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Réplica das figuras de estatísticas, mas removendo a terceira coluna
(Correlation e Index of Agreement)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Diretórios
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "summary_statistics.csv"
STATIONS_FILE = SCRIPT_DIR.parent / "stations_match.csv"
OUTPUT_DIR = SCRIPT_DIR

def create_general_statistics_figure():
    """Cria figura das estatísticas gerais (sem correlação e index of agreement)"""
    
    # Lê dados
    df = pd.read_csv(CSV_FILE)
    df_stations = pd.read_csv(STATIONS_FILE)
    
    # Merge para ter coordenadas
    df = pd.merge(df, df_stations[['estacao', 'lon_obs', 'lat_obs']], 
                  on='estacao', how='left')
    
    # Identifica tipo de estação
    df['tipo'] = df['estacao'].apply(
        lambda x: 'EMODNET' if str(x).startswith(('6', '13', '55', '56', '66')) else 'NDBC'
    )
    
    # Remove estações com valores infinitos
    df_clean = df[np.isfinite(df['rmse']) & np.isfinite(df['bias']) & 
                  np.isfinite(df['mae']) & np.isfinite(df['si'])]
    
    # Calcula NBIAS e NRMSE normalizados
    df_clean['nbias'] = (df_clean['bias'] / df_clean['hs_obs_mean']) * 100
    df_clean['nrmse'] = (df_clean['rmse'] / df_clean['hs_obs_mean']) * 100
    
    # Remove outliers extremos
    df_clean = df_clean[
        (df_clean['nbias'] >= -100) & (df_clean['nbias'] <= 100) &
        (df_clean['nrmse'] >= 0) & (df_clean['nrmse'] <= 100) &
        (df_clean['mae'] >= 0) & (df_clean['mae'] <= 2) &
        (df_clean['si'] >= 0) & (df_clean['si'] <= 1)
    ]
    
    # Cores
    colors = {'NDBC': '#6FAADB', 'EMODNET': '#D9534F'}
    
    # Cria figura com layout 2x2
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Histograma de NBIAS
    ax1 = fig.add_subplot(gs[0, 0])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['nbias']
        ax1.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='NBias = 0')
    ax1.set_xlabel('NBIAS (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('NBIAS Distribution')
    # Limite automático
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de NRMSE
    ax2 = fig.add_subplot(gs[0, 1])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['nrmse']
        ax2.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax2.set_xlabel('NRMSE (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('NRMSE Distribution')
    # Limite automático
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax2.grid(True, alpha=0.3)
    
    # 3. Histograma de MAE
    ax3 = fig.add_subplot(gs[1, 0])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['mae']
        ax3.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax3.set_xlabel('MAE (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('MAE Distribution')
    # Limite automático
    ax3.legend(fontsize=10)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax3.grid(True, alpha=0.3)
    
    # 4. Histograma de SI
    ax4 = fig.add_subplot(gs[1, 1])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['si']
        ax4.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax4.set_xlabel('SI (Scatter Index)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('SI Distribution')
    # Limite automático
    ax4.legend(fontsize=10)
    ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax4.grid(True, alpha=0.3)
    
    # Título geral
    fig.suptitle('WW3 Model Validation - Normalized Metrics', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Salva figura
    output_file = OUTPUT_DIR / "statistics_summary_no_corr.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Figura geral salva: {output_file}")
    plt.close()

def create_p99_statistics_figure():
    """Cria figura das estatísticas P99 (sem correlação e index of agreement)"""
    
    # Lê dados
    df = pd.read_csv(CSV_FILE)
    df_stations = pd.read_csv(STATIONS_FILE)
    
    # Merge para ter coordenadas
    df = pd.merge(df, df_stations[['estacao', 'lon_obs', 'lat_obs']], 
                  on='estacao', how='left')
    
    # Identifica tipo de estação
    df['tipo'] = df['estacao'].apply(
        lambda x: 'EMODNET' if str(x).startswith(('6', '13', '55', '56', '66')) else 'NDBC'
    )
    
    # Remove estações com valores infinitos
    df_clean = df[np.isfinite(df['rmse_p99']) & np.isfinite(df['bias_p99']) & 
                  np.isfinite(df['mae_p99']) & np.isfinite(df['si_p99'])]
    
    # Calcula NBIAS e NRMSE P99 normalizados
    df_clean['nbias_p99'] = (df_clean['bias_p99'] / df_clean['p99_obs']) * 100
    df_clean['nrmse_p99'] = (df_clean['rmse_p99'] / df_clean['p99_obs']) * 100
    
    # Remove outliers extremos para P99
    df_clean = df_clean[
        (df_clean['nbias_p99'] >= -100) & (df_clean['nbias_p99'] <= 100) &
        (df_clean['nrmse_p99'] >= 0) & (df_clean['nrmse_p99'] <= 100) &
        (df_clean['mae_p99'] >= 0) & (df_clean['mae_p99'] <= 3) &
        (df_clean['si_p99'] >= 0) & (df_clean['si_p99'] <= 1)
    ]
    
    # Cores
    colors = {'NDBC': '#6FAADB', 'EMODNET': '#D9534F'}
    
    # Cria figura com layout 2x2
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Histograma de NBIAS P99
    ax1 = fig.add_subplot(gs[0, 0])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['nbias_p99']
        ax1.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='NBias = 0')
    ax1.set_xlabel('NBIAS P99 (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('NBIAS Distribution (P99)')
    # Limite automático
    ax1.legend(fontsize=10)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de NRMSE P99
    ax2 = fig.add_subplot(gs[0, 1])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['nrmse_p99']
        ax2.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax2.set_xlabel('NRMSE P99 (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('NRMSE Distribution (P99)')
    # Limite automático
    ax2.legend(fontsize=10)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax2.grid(True, alpha=0.3)
    
    # 3. Histograma de MAE P99
    ax3 = fig.add_subplot(gs[1, 0])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['mae_p99']
        ax3.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax3.set_xlabel('MAE P99 (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('MAE Distribution (P99)')
    # Limite automático
    ax3.legend(fontsize=10)
    ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax3.grid(True, alpha=0.3)
    
    # 4. Histograma de SI P99
    ax4 = fig.add_subplot(gs[1, 1])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['si_p99']
        ax4.hist(data, bins=30, alpha=0.7, label=tipo, color=colors[tipo], edgecolor='black')
    ax4.set_xlabel('SI P99 (Scatter Index)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('SI Distribution (P99)')
    # Limite automático
    ax4.legend(fontsize=10)
    ax4.yaxis.set_major_locator(plt.MaxNLocator(integer=True, prune=None, nbins=8))
    ax4.grid(True, alpha=0.3)
    
    # Título geral
    fig.suptitle('WW3 Model Validation - 99th Percentile Analysis (Extreme Events)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Salva figura
    output_file = OUTPUT_DIR / "statistics_p99_no_corr.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Figura P99 salva: {output_file}")
    plt.close()

def main():
    """Função principal"""
    print("="*70)
    print("CRIANDO FIGURAS SEM CORRELAÇÃO E INDEX OF AGREEMENT")
    print("="*70)
    
    # Cria figura das estatísticas gerais
    print("\n📈 Criando figura das estatísticas gerais...")
    create_general_statistics_figure()
    
    # Cria figura das estatísticas P99
    print("📈 Criando figura das estatísticas P99...")
    create_p99_statistics_figure()
    
    print("\n🎯 Duas figuras criadas com sucesso!")
    print("📁 Arquivos gerados:")
    print("   - statistics_summary_no_corr.png (estatísticas gerais)")
    print("   - statistics_p99_no_corr.png (estatísticas P99)")
    print("="*70)

if __name__ == "__main__":
    main()