#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualização das estatísticas de validação modelo vs observações
Com NRMSE e NBIAS normalizados
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
OUTPUT_FILE = OUTPUT_DIR / "statistics_summary_plot.png"

def main():
    """Função principal"""
    print("="*70)
    print("VISUALIZAÇÃO DAS ESTATÍSTICAS - MODELO vs OBSERVAÇÕES")
    print("="*70)
    
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
    df_clean = df[
        ~df['bias'].isin([np.inf, -np.inf]) & 
        ~df['rmse'].isin([np.inf, -np.inf]) &
        ~df['lon_obs'].isin([np.inf, -np.inf]) &
        ~df['lat_obs'].isin([np.inf, -np.inf]) &
        ~df['correlacao'].isna()
    ].copy()
    
    # Calcula NRMSE e NBIAS normalizados em percentual
    df_clean['nrmse'] = (df_clean['rmse'] / df_clean['hs_obs_mean']) * 100
    df_clean['nbias'] = (df_clean['bias'] / df_clean['hs_obs_mean']) * 100
    
    print(f"\n📊 Dados carregados:")
    print(f"   Total de estações: {len(df)}")
    print(f"   Estações válidas: {len(df_clean)}")
    print(f"   - NDBC: {len(df_clean[df_clean['tipo']=='NDBC'])}")
    print(f"   - EMODNET: {len(df_clean[df_clean['tipo']=='EMODNET'])}")
    
    # Cria figura com 2 linhas x 3 colunas
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Define cores
    colors = {'NDBC': '#2E86AB', 'EMODNET': '#A23B72'}
    
    # =====================================================================
    # LINHA SUPERIOR: DISTRIBUIÇÕES
    # =====================================================================
    
    # 1. Histograma de NBIAS
    ax1 = fig.add_subplot(gs[0, 0])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['nbias']
        ax1.hist(data, bins=30, alpha=0.6, label=tipo, color=colors[tipo], edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='NBias = 0')
    ax1.set_xlabel('NBIAS (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('NBIAS Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograma de NRMSE
    ax2 = fig.add_subplot(gs[0, 1])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['nrmse']
        ax2.hist(data, bins=30, alpha=0.6, label=tipo, color=colors[tipo], edgecolor='black')
    ax2.set_xlabel('NRMSE (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('NRMSE Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Histograma de Correlação
    ax3 = fig.add_subplot(gs[0, 2])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['correlacao']
        ax3.hist(data, bins=30, alpha=0.6, label=tipo, color=colors[tipo], edgecolor='black')
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Correlation Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # =====================================================================
    # LINHA INFERIOR: MAIS DISTRIBUIÇÕES
    # =====================================================================
    
    # 4. Histograma de MAE
    ax4 = fig.add_subplot(gs[1, 0])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['mae']
        ax4.hist(data, bins=30, alpha=0.6, label=tipo, color=colors[tipo], edgecolor='black')
    ax4.set_xlabel('MAE (m)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('MAE Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Histograma de SI (Scatter Index)
    ax5 = fig.add_subplot(gs[1, 1])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['si']
        ax5.hist(data, bins=30, alpha=0.6, label=tipo, color=colors[tipo], edgecolor='black')
    ax5.set_xlabel('SI (Scatter Index)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('SI Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Histograma de Index of Agreement
    ax6 = fig.add_subplot(gs[1, 2])
    for tipo in ['NDBC', 'EMODNET']:
        data = df_clean[df_clean['tipo']==tipo]['index_agreement']
        ax6.hist(data, bins=30, alpha=0.6, label=tipo, color=colors[tipo], edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='d = 0 (no agreement)')
    ax6.axvline(1, color='green', linestyle='--', linewidth=2, label='d = 1 (perfect)')
    ax6.set_xlabel('Index of Agreement (d)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Index of Agreement Distribution')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Título geral
    fig.suptitle('WW3 Model Validation - Normalized Metrics', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Salva figura
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figura salva: {OUTPUT_FILE}")
    
    # Mostra estatísticas no terminal
    print("\n" + "="*70)
    print("📊 ESTATÍSTICAS POR TIPO DE ESTAÇÃO:")
    print("="*70)
    
    # Calcula estatísticas
    stats_ndbc = df_clean[df_clean['tipo']=='NDBC'][['nbias', 'nrmse', 'correlacao', 'mae', 'si', 'index_agreement']].describe()
    stats_emodnet = df_clean[df_clean['tipo']=='EMODNET'][['nbias', 'nrmse', 'correlacao', 'mae', 'si', 'index_agreement']].describe()
    
    print("\n🔵 NDBC:")
    print(f"   N estações:     {len(df_clean[df_clean['tipo']=='NDBC'])}")
    print(f"   NBIAS:          {stats_ndbc.loc['mean', 'nbias']:7.2f} ± {stats_ndbc.loc['std', 'nbias']:.2f} %")
    print(f"   NRMSE:          {stats_ndbc.loc['mean', 'nrmse']:7.2f} ± {stats_ndbc.loc['std', 'nrmse']:.2f} %")
    print(f"   Correlação:     {stats_ndbc.loc['mean', 'correlacao']:7.3f} ± {stats_ndbc.loc['std', 'correlacao']:.3f}")
    print(f"   MAE:            {stats_ndbc.loc['mean', 'mae']:7.3f} ± {stats_ndbc.loc['std', 'mae']:.3f} m")
    print(f"   SI:             {stats_ndbc.loc['mean', 'si']:7.3f} ± {stats_ndbc.loc['std', 'si']:.3f}")
    print(f"   Index of Agr:   {stats_ndbc.loc['mean', 'index_agreement']:7.3f} ± {stats_ndbc.loc['std', 'index_agreement']:.3f}")
    
    print("\n🟣 EMODNET:")
    print(f"   N estações:     {len(df_clean[df_clean['tipo']=='EMODNET'])}")
    print(f"   NBIAS:          {stats_emodnet.loc['mean', 'nbias']:7.2f} ± {stats_emodnet.loc['std', 'nbias']:.2f} %")
    print(f"   NRMSE:          {stats_emodnet.loc['mean', 'nrmse']:7.2f} ± {stats_emodnet.loc['std', 'nrmse']:.2f} %")
    print(f"   Correlação:     {stats_emodnet.loc['mean', 'correlacao']:7.3f} ± {stats_emodnet.loc['std', 'correlacao']:.3f}")
    print(f"   MAE:            {stats_emodnet.loc['mean', 'mae']:7.3f} ± {stats_emodnet.loc['std', 'mae']:.3f} m")
    print(f"   SI:             {stats_emodnet.loc['mean', 'si']:7.3f} ± {stats_emodnet.loc['std', 'si']:.3f}")
    print(f"   Index of Agr:   {stats_emodnet.loc['mean', 'index_agreement']:7.3f} ± {stats_emodnet.loc['std', 'index_agreement']:.3f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
