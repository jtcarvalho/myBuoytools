#!/usr/bin/env python3
"""
Script para criar figura combinada das distribuições estatísticas - Versão limpa com box plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_clean_distributions():
    """
    Cria figura combinada com distribuições mais limpas usando box plots
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
    
    # Preparar dados para plotting
    data_for_plot = []
    
    metrics_info = [
        ('nbias_pct', 'nbias_p99_pct', 'NBIAS (%)'),
        ('nrmse_pct', 'nrmse_p99_pct', 'NRMSE (%)'),
        ('mae', 'mae_p99', 'MAE (m)'),
        ('si', 'si_p99', 'SI')
    ]
    
    # Reformatar dados para seaborn
    for metric_gen, metric_p99, metric_name in metrics_info:
        # General
        for _, row in df.iterrows():
            data_for_plot.append({
                'metric': metric_name,
                'type': row['tipo_estacao'],
                'period': 'General',
                'value': row[metric_gen],
                'category': f"{row['tipo_estacao']}_General"
            })
        # P99
        for _, row in df.iterrows():
            data_for_plot.append({
                'metric': metric_name,
                'type': row['tipo_estacao'],
                'period': 'P99',
                'value': row[metric_p99],
                'category': f"{row['tipo_estacao']}_P99"
            })
    
    plot_df = pd.DataFrame(data_for_plot)
    
    # Criar figura com 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('WW3 Model Validation - General vs P99 Statistics Comparison', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Definir cores mais suaves
    palette = {
        'NDBC_General': '#87CEEB',     # Azul claro
        'NDBC_P99': '#4682B4',         # Azul médio
        'EMODNET_General': '#DDA0DD',  # Violeta claro
        'EMODNET_P99': '#9370DB'       # Violeta médio
    }
    
    metrics_plot = ['NBIAS (%)', 'NRMSE (%)', 'MAE (m)', 'SI']
    panel_labels = ['a', 'b', 'c', 'd']
    
    for i, (metric_name, panel_label) in enumerate(zip(metrics_plot, panel_labels)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Filtrar dados para esta métrica
        metric_data = plot_df[plot_df['metric'] == metric_name]
        
        # Criar box plot
        sns.boxplot(data=metric_data, x='type', y='value', hue='period', 
                   ax=ax, palette=['#87CEEB', '#4682B4', '#DDA0DD', '#9370DB'])
        
        # Personalizar
        ax.set_xlabel('Station Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Adicionar linha de referência para NBIAS
        if 'NBIAS' in metric_name:
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # Adicionar identificação do painel
        ax.text(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
               fontsize=14, fontweight='bold', verticalalignment='top')
        
        # Personalizar legenda
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['General', 'P99'], loc='upper right', fontsize=11)
        else:
            ax.get_legend().remove()
        
        # Adicionar médias como texto
        ndbc_gen = metric_data[(metric_data['type'] == 'NDBC') & (metric_data['period'] == 'General')]['value'].mean()
        ndbc_p99 = metric_data[(metric_data['type'] == 'NDBC') & (metric_data['period'] == 'P99')]['value'].mean()
        emodnet_gen = metric_data[(metric_data['type'] == 'EMODNET') & (metric_data['period'] == 'General')]['value'].mean()
        emodnet_p99 = metric_data[(metric_data['type'] == 'EMODNET') & (metric_data['period'] == 'P99')]['value'].mean()
        
        stats_text = [
            f'NDBC: {ndbc_gen:.2f} | {ndbc_p99:.2f}',
            f'EMODNET: {emodnet_gen:.2f} | {emodnet_p99:.2f}'
        ]
        
        ax.text(0.98, 0.02, '\n'.join(stats_text), transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9),
               fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    # Salvar figura
    output_file = os.path.join(script_dir, 'clean_distributions_general_vs_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figura limpa salva: {output_file}")
    
    plt.show()

def create_violin_plot_version():
    """
    Versão alternativa com violin plots
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    
    # Ler dados
    df = pd.read_csv(stats_file)
    
    # Calcular métricas
    df['nrmse_pct'] = df['si'] * 100
    df['nbias_pct'] = (df['bias'] / df['hs_obs_mean']) * 100
    df['nbias_p99_pct'] = (df['bias_p99'] / df['p99_obs']) * 100
    df['nrmse_p99_pct'] = (df['rmse_p99'] / df['p99_obs']) * 100
    df['si_p99'] = df['rmse_p99'] / df['p99_obs']
    df['tipo_estacao'] = df['estacao'].apply(lambda x: 'NDBC' if str(x).startswith(('4', '5')) else 'EMODNET')
    
    # Preparar dados
    data_for_plot = []
    metrics_info = [
        ('nbias_pct', 'nbias_p99_pct', 'NBIAS (%)'),
        ('nrmse_pct', 'nrmse_p99_pct', 'NRMSE (%)'),
        ('mae', 'mae_p99', 'MAE (m)'),
        ('si', 'si_p99', 'SI')
    ]
    
    for metric_gen, metric_p99, metric_name in metrics_info:
        for _, row in df.iterrows():
            data_for_plot.append({
                'metric': metric_name,
                'type': row['tipo_estacao'],
                'period': 'General',
                'value': row[metric_gen]
            })
            data_for_plot.append({
                'metric': metric_name,
                'type': row['tipo_estacao'],
                'period': 'P99',
                'value': row[metric_p99]
            })
    
    plot_df = pd.DataFrame(data_for_plot)
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('WW3 Model Validation - General vs P99 Statistics (Violin Plots)', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    metrics_plot = ['NBIAS (%)', 'NRMSE (%)', 'MAE (m)', 'SI']
    panel_labels = ['a', 'b', 'c', 'd']
    
    for i, (metric_name, panel_label) in enumerate(zip(metrics_plot, panel_labels)):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        metric_data = plot_df[plot_df['metric'] == metric_name]
        
        # Violin plot
        sns.violinplot(data=metric_data, x='type', y='value', hue='period', 
                      ax=ax, palette=['#87CEEB', '#4682B4', '#DDA0DD', '#9370DB'],
                      split=False, inner='box')
        
        ax.set_xlabel('Station Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if 'NBIAS' in metric_name:
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax.text(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
               fontsize=14, fontweight='bold', verticalalignment='top')
        
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, ['General', 'P99'], loc='upper right', fontsize=11)
        else:
            ax.get_legend().remove()
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    output_file = os.path.join(script_dir, 'violin_distributions_general_vs_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figura violin salva: {output_file}")
    
    plt.show()

if __name__ == "__main__":
    print("Criando versão com box plots...")
    create_clean_distributions()
    
    print("\nCriando versão com violin plots...")
    create_violin_plot_version()