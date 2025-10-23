#!/usr/bin/env python3
"""
Script para criar figura combinada das distribuições estatísticas - Geral vs P99
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_combined_distributions():
    """
    Cria figura combinada com distribuições das estatísticas gerais vs P99
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    
    if not os.path.exists(stats_file):
        print(f"Erro: Arquivo {stats_file} não encontrado!")
        return
    
    # Ler dados
    df = pd.read_csv(stats_file)
    print(f"📊 Dados carregados: {len(df)} estações")
    
    # Calcular métricas
    df['nrmse_pct'] = df['si'] * 100
    df['nbias_pct'] = (df['bias'] / df['hs_obs_mean']) * 100
    df['nbias_p99_pct'] = (df['bias_p99'] / df['p99_obs']) * 100
    df['nrmse_p99_pct'] = (df['rmse_p99'] / df['p99_obs']) * 100
    df['si_p99'] = df['rmse_p99'] / df['p99_obs']
    
    # Classificar por tipo de estação
    df['tipo_estacao'] = df['estacao'].apply(lambda x: 'NDBC' if str(x).startswith(('4', '5')) else 'EMODNET')
    
    # Separar dados por tipo
    ndbc_data = df[df['tipo_estacao'] == 'NDBC']
    emodnet_data = df[df['tipo_estacao'] == 'EMODNET']
    
    print(f"🔵 NDBC: {len(ndbc_data)} estações")
    print(f"🟣 EMODNET: {len(emodnet_data)} estações")
    
    # Definir cores
    colors = {
        'ndbc_general': '#87CEEB',      # Azul claro
        'ndbc_p99': '#1E90FF',          # Azul escuro
        'emodnet_general': '#DDA0DD',   # Violeta claro
        'emodnet_p99': '#8B008B'        # Violeta escuro
    }
    
    # Criar figura com 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    fig.suptitle('WW3 Model Validation - General vs P99 Statistics Distributions', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Configurar métricas
    metrics = [
        ('nbias_pct', 'nbias_p99_pct', 'NBIAS (%)', (-40, 40), 'a'),
        ('nrmse_pct', 'nrmse_p99_pct', 'NRMSE (%)', (0, 60), 'b'),
        ('mae', 'mae_p99', 'MAE (m)', (0, 2.0), 'c'),
        ('si', 'si_p99', 'SI (Scatter Index)', (0, 0.7), 'd')
    ]
    
    for i, (metric_gen, metric_p99, title, xlim, panel_label) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Plotar histogramas
        # NDBC General (azul claro)
        ax.hist(ndbc_data[metric_gen], bins=20, alpha=0.7, 
               color=colors['ndbc_general'], edgecolor='black', linewidth=0.5,
               label=f'NDBC General ({len(ndbc_data)})', density=True)
        
        # NDBC P99 (azul escuro)
        ax.hist(ndbc_data[metric_p99], bins=20, alpha=0.7, 
               color=colors['ndbc_p99'], edgecolor='black', linewidth=0.5,
               label=f'NDBC P99 ({len(ndbc_data)})', density=True)
        
        # EMODNET General (violeta claro)
        ax.hist(emodnet_data[metric_gen], bins=20, alpha=0.7, 
               color=colors['emodnet_general'], edgecolor='black', linewidth=0.5,
               label=f'EMODNET General ({len(emodnet_data)})', density=True)
        
        # EMODNET P99 (violeta escuro)
        ax.hist(emodnet_data[metric_p99], bins=20, alpha=0.7, 
               color=colors['emodnet_p99'], edgecolor='black', linewidth=0.5,
               label=f'EMODNET P99 ({len(emodnet_data)})', density=True)
        
        # Adicionar linha de referência para NBIAS
        if 'nbias' in metric_gen.lower():
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='NBias = 0')
        
        # Configurar eixos
        ax.set_xlabel(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_xlim(xlim)
        ax.grid(True, alpha=0.3)
        
        # Adicionar identificação do painel
        ax.text(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
               fontsize=14, fontweight='bold', verticalalignment='top')
        
        # Adicionar legenda apenas no primeiro subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=10)
        
        # Adicionar estatísticas médias como texto
        stats_text = []
        
        # NDBC
        ndbc_gen_mean = ndbc_data[metric_gen].mean()
        ndbc_p99_mean = ndbc_data[metric_p99].mean()
        stats_text.append(f'NDBC Gen: {ndbc_gen_mean:.2f}')
        stats_text.append(f'NDBC P99: {ndbc_p99_mean:.2f}')
        
        # EMODNET  
        emodnet_gen_mean = emodnet_data[metric_gen].mean()
        emodnet_p99_mean = emodnet_data[metric_p99].mean()
        stats_text.append(f'EMODNET Gen: {emodnet_gen_mean:.2f}')
        stats_text.append(f'EMODNET P99: {emodnet_p99_mean:.2f}')
        
        # Posicionar texto de estatísticas
        ax.text(0.98, 0.98, '\n'.join(stats_text), transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
               fontsize=9, verticalalignment='top', horizontalalignment='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Salvar figura
    output_file = os.path.join(script_dir, 'combined_distributions_general_vs_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figura combinada salva: {output_file}")
    
    # Imprimir resumo estatístico
    print("\n📊 RESUMO ESTATÍSTICO:")
    print("="*80)
    
    metrics_summary = [
        ('NBIAS (%)', 'nbias_pct', 'nbias_p99_pct'),
        ('NRMSE (%)', 'nrmse_pct', 'nrmse_p99_pct'),
        ('MAE (m)', 'mae', 'mae_p99'),
        ('SI', 'si', 'si_p99')
    ]
    
    for metric_name, metric_gen, metric_p99 in metrics_summary:
        print(f"\n{metric_name}:")
        print(f"  🔵 NDBC General:     {ndbc_data[metric_gen].mean():.3f} ± {ndbc_data[metric_gen].std():.3f}")
        print(f"  🔵 NDBC P99:         {ndbc_data[metric_p99].mean():.3f} ± {ndbc_data[metric_p99].std():.3f}")
        print(f"  🟣 EMODNET General:  {emodnet_data[metric_gen].mean():.3f} ± {emodnet_data[metric_gen].std():.3f}")
        print(f"  🟣 EMODNET P99:      {emodnet_data[metric_p99].mean():.3f} ± {emodnet_data[metric_p99].std():.3f}")
    
    plt.show()

if __name__ == "__main__":
    create_combined_distributions()