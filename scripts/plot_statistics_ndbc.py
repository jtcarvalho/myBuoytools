#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualização resumida das estatísticas - apenas estações NDBC válidas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuração
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11

SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "summary_statistics.csv"
OUTPUT_FILE = SCRIPT_DIR / "statistics_ndbc_summary.png"

def main():
    print("="*70)
    print("RESUMO ESTATÍSTICO - ESTAÇÕES NDBC")
    print("="*70)
    
    # Lê dados
    df = pd.read_csv(CSV_FILE)
    
    # Filtra apenas NDBC válidas
    df_ndbc = df[
        ~df['estacao'].astype(str).str.startswith(('6', '13', '55', '56', '66')) &
        ~df['bias'].isin([np.inf, -np.inf]) & 
        ~df['rmse'].isin([np.inf, -np.inf])
    ].copy()
    
    print(f"\n📊 Total de estações NDBC válidas: {len(df_ndbc)}")
    
    # Cria figura
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Validação Modelo WW3 - Estações NDBC (N=139)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    color_main = '#2E86AB'
    
    # 1. Histograma Bias
    ax = axes[0, 0]
    ax.hist(df_ndbc['bias'], bins=30, color=color_main, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Bias = 0')
    ax.axvline(df_ndbc['bias'].mean(), color='orange', linestyle='-', linewidth=2, 
               label=f'Média = {df_ndbc["bias"].mean():.3f} m')
    ax.set_xlabel('Bias (m)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição do Bias')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 2. Histograma RMSE
    ax = axes[0, 1]
    ax.hist(df_ndbc['rmse'], bins=30, color=color_main, alpha=0.7, edgecolor='black')
    ax.axvline(df_ndbc['rmse'].mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Média = {df_ndbc["rmse"].mean():.3f} m')
    ax.set_xlabel('RMSE (m)')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição do RMSE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 3. Histograma Correlação
    ax = axes[0, 2]
    ax.hist(df_ndbc['correlacao'], bins=30, color=color_main, alpha=0.7, edgecolor='black')
    ax.axvline(df_ndbc['correlacao'].mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Média = {df_ndbc["correlacao"].mean():.3f}')
    ax.set_xlabel('Correlação')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição da Correlação')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. Scatter: RMSE vs Correlação
    ax = axes[1, 0]
    scatter = ax.scatter(df_ndbc['correlacao'], df_ndbc['rmse'], 
                        c=df_ndbc['bias'], cmap='RdBu_r', s=60, 
                        alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Correlação')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE vs Correlação')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Bias (m)', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 5. Scatter: Bias vs RMSE  
    ax = axes[1, 1]
    ax.scatter(df_ndbc['bias'], df_ndbc['rmse'], 
              c=df_ndbc['correlacao'], cmap='viridis', s=60,
              alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_xlabel('Bias (m)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE vs Bias')
    ax.grid(True, alpha=0.3)
    
    # 6. Tabela resumo
    ax = axes[1, 2]
    ax.axis('off')
    
    stats = df_ndbc[['bias', 'rmse', 'mae', 'correlacao', 'n_pontos']].describe()
    
    table_text = f"""
╔═══════════════════════════════════╗
║   ESTATÍSTICAS RESUMO - NDBC      ║
╠═══════════════════════════════════╣
║                                   ║
║  Estações: {len(df_ndbc):3d}                   ║
║                                   ║
║  BIAS (m)                         ║
║    Média:   {stats.loc['mean', 'bias']:7.3f} ± {stats.loc['std', 'bias']:.3f}      ║
║    Mediana: {stats.loc['50%', 'bias']:7.3f}                ║
║    Min/Max: {stats.loc['min', 'bias']:7.3f} / {stats.loc['max', 'bias']:6.3f}      ║
║                                   ║
║  RMSE (m)                         ║
║    Média:   {stats.loc['mean', 'rmse']:7.3f} ± {stats.loc['std', 'rmse']:.3f}      ║
║    Mediana: {stats.loc['50%', 'rmse']:7.3f}                ║
║    Min/Max: {stats.loc['min', 'rmse']:7.3f} / {stats.loc['max', 'rmse']:6.3f}      ║
║                                   ║
║  MAE (m)                          ║
║    Média:   {stats.loc['mean', 'mae']:7.3f} ± {stats.loc['std', 'mae']:.3f}      ║
║    Mediana: {stats.loc['50%', 'mae']:7.3f}                ║
║                                   ║
║  CORRELAÇÃO                       ║
║    Média:   {stats.loc['mean', 'correlacao']:7.3f} ± {stats.loc['std', 'correlacao']:.3f}      ║
║    Mediana: {stats.loc['50%', 'correlacao']:7.3f}                ║
║    Min/Max: {stats.loc['min', 'correlacao']:7.3f} / {stats.loc['max', 'correlacao']:6.3f}      ║
║                                   ║
║  N PONTOS (média)                 ║
║    Média:   {stats.loc['mean', 'n_pontos']:7.0f}                ║
║                                   ║
╚═══════════════════════════════════╝
"""
    
    ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Figura salva: {OUTPUT_FILE}")
    
    # Estatísticas no terminal
    print("\n" + "="*70)
    print("📊 ESTATÍSTICAS DETALHADAS:")
    print("="*70)
    print(f"\n{'Métrica':<15} {'Média':<10} {'Desvio':<10} {'Min':<10} {'Max':<10}")
    print("-"*70)
    print(f"{'Bias (m)':<15} {stats.loc['mean', 'bias']:>9.3f} {stats.loc['std', 'bias']:>9.3f} "
          f"{stats.loc['min', 'bias']:>9.3f} {stats.loc['max', 'bias']:>9.3f}")
    print(f"{'RMSE (m)':<15} {stats.loc['mean', 'rmse']:>9.3f} {stats.loc['std', 'rmse']:>9.3f} "
          f"{stats.loc['min', 'rmse']:>9.3f} {stats.loc['max', 'rmse']:>9.3f}")
    print(f"{'MAE (m)':<15} {stats.loc['mean', 'mae']:>9.3f} {stats.loc['std', 'mae']:>9.3f} "
          f"{stats.loc['min', 'mae']:>9.3f} {stats.loc['max', 'mae']:>9.3f}")
    print(f"{'Correlação':<15} {stats.loc['mean', 'correlacao']:>9.3f} {stats.loc['std', 'correlacao']:>9.3f} "
          f"{stats.loc['min', 'correlacao']:>9.3f} {stats.loc['max', 'correlacao']:>9.3f}")
    print(f"{'N pontos':<15} {stats.loc['mean', 'n_pontos']:>9.0f} {stats.loc['std', 'n_pontos']:>9.0f} "
          f"{stats.loc['min', 'n_pontos']:>9.0f} {stats.loc['max', 'n_pontos']:>9.0f}")
    print("="*70)
    
    # Performance summary
    print("\n🎯 RESUMO DE PERFORMANCE:")
    print(f"   • Modelo tem viés pequeno: {abs(stats.loc['mean', 'bias']):.3f} m")
    print(f"   • Erro típico (RMSE): {stats.loc['mean', 'rmse']:.3f} m")
    print(f"   • Correlação excelente: {stats.loc['mean', 'correlacao']:.3f}")
    
    # Classifica estações
    excellent = len(df_ndbc[df_ndbc['correlacao'] >= 0.95])
    good = len(df_ndbc[(df_ndbc['correlacao'] >= 0.85) & (df_ndbc['correlacao'] < 0.95)])
    fair = len(df_ndbc[(df_ndbc['correlacao'] >= 0.70) & (df_ndbc['correlacao'] < 0.85)])
    poor = len(df_ndbc[df_ndbc['correlacao'] < 0.70])
    
    print(f"\n📈 DISTRIBUIÇÃO DE QUALIDADE (por correlação):")
    print(f"   • Excelente (R ≥ 0.95): {excellent:3d} ({100*excellent/len(df_ndbc):5.1f}%)")
    print(f"   • Boa (0.85 ≤ R < 0.95): {good:3d} ({100*good/len(df_ndbc):5.1f}%)")
    print(f"   • Razoável (0.70 ≤ R < 0.85): {fair:3d} ({100*fair/len(df_ndbc):5.1f}%)")
    print(f"   • Fraca (R < 0.70):     {poor:3d} ({100*poor/len(df_ndbc):5.1f}%)")
    print("="*70)

if __name__ == "__main__":
    main()
