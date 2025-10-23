#!/usr/bin/env python3
"""
Script para criar figura super limpa com barras de erro - comparação simples e clara
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def create_simple_comparison():
    """
    Cria figura simples com barras de erro mostrando médias e desvios padrão
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
    
    # Separar dados
    ndbc_data = df[df['tipo_estacao'] == 'NDBC']
    emodnet_data = df[df['tipo_estacao'] == 'EMODNET']
    
    # Criar figura
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('WW3 Model Validation - General vs P99 Statistics', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Definir métricas
    metrics = [
        ('nbias_pct', 'nbias_p99_pct', 'NBIAS (%)', 'a'),
        ('nrmse_pct', 'nrmse_p99_pct', 'NRMSE (%)', 'b'),
        ('mae', 'mae_p99', 'MAE (m)', 'c'),
        ('si', 'si_p99', 'SI', 'd')
    ]
    
    # Cores mais simples
    colors = {
        'ndbc_gen': '#5B9BD5',    # Azul
        'ndbc_p99': '#2F5F8F',    # Azul escuro
        'emodnet_gen': '#A5A5A5', # Cinza claro
        'emodnet_p99': '#595959'  # Cinza escuro
    }
    
    for i, (metric_gen, metric_p99, title, panel_label) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        # Calcular estatísticas
        stats = {
            'NDBC General': (ndbc_data[metric_gen].mean(), ndbc_data[metric_gen].std()),
            'NDBC P99': (ndbc_data[metric_p99].mean(), ndbc_data[metric_p99].std()),
            'EMODNET General': (emodnet_data[metric_gen].mean(), emodnet_data[metric_gen].std()),
            'EMODNET P99': (emodnet_data[metric_p99].mean(), emodnet_data[metric_p99].std())
        }
        
        # Posições das barras
        x = np.arange(2)  # NDBC, EMODNET
        width = 0.35
        
        # Dados para plotting
        ndbc_means = [stats['NDBC General'][0], stats['NDBC P99'][0]]
        ndbc_stds = [stats['NDBC General'][1], stats['NDBC P99'][1]]
        emodnet_means = [stats['EMODNET General'][0], stats['EMODNET P99'][0]]
        emodnet_stds = [stats['EMODNET General'][1], stats['EMODNET P99'][1]]
        
        # Plotar barras
        bars1 = ax.bar(x - width/2, [ndbc_means[0], emodnet_means[0]], width, 
                      yerr=[ndbc_stds[0], emodnet_stds[0]], 
                      label='General', color=[colors['ndbc_gen'], colors['emodnet_gen']],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        bars2 = ax.bar(x + width/2, [ndbc_means[1], emodnet_means[1]], width,
                      yerr=[ndbc_stds[1], emodnet_stds[1]], 
                      label='P99', color=[colors['ndbc_p99'], colors['emodnet_p99']],
                      capsize=5, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Personalizar
        ax.set_xlabel('Station Type', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['NDBC', 'EMODNET'])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Linha de referência para NBIAS
        if 'NBIAS' in title:
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        # Ajustar limites do eixo Y para dar espaço aos valores
        y_min, y_max = ax.get_ylim()
        if 'NBIAS' in title:
            # Para NBIAS, que pode ter valores negativos
            ax.set_ylim(y_min - abs(y_max - y_min) * 0.15, y_max + abs(y_max - y_min) * 0.15)
        else:
            # Para outras métricas (sempre positivas)
            ax.set_ylim(0, y_max + abs(y_max) * 0.15)
        
        # Painel label
        ax.text(0.02, 0.98, f'({panel_label})', transform=ax.transAxes,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
               fontsize=14, fontweight='bold', verticalalignment='top')
        
        # Legenda apenas no primeiro subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=11)
        
        # Adicionar valores nas barras com posicionamento melhorado
        for j, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            std1 = ndbc_stds[j] if j == 0 else emodnet_stds[j]
            std2 = ndbc_stds[j] if j == 0 else emodnet_stds[j]
            
            # Para valores negativos, colocar o texto abaixo da barra
            if height1 < 0:
                y_pos1 = height1 - std1 - 0.5  # Abaixo da barra com margem
                va1 = 'top'
            else:
                y_pos1 = height1 + std1 + 0.5  # Acima da barra com margem
                va1 = 'bottom'
                
            if height2 < 0:
                y_pos2 = height2 - std2 - 0.5  # Abaixo da barra com margem
                va2 = 'top'
            else:
                y_pos2 = height2 + std2 + 0.5  # Acima da barra com margem
                va2 = 'bottom'
            
            # Texto com posicionamento ajustado
            ax.text(bar1.get_x() + bar1.get_width()/2., y_pos1,
                   f'{height1:.1f}', ha='center', va=va1, fontsize=9, fontweight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., y_pos2,
                   f'{height2:.1f}', ha='center', va=va2, fontsize=9, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    # Salvar
    output_file = os.path.join(script_dir, 'simple_comparison_general_vs_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figura simples salva: {output_file}")
    
    # Criar também uma versão com dados separados lado a lado
    create_side_by_side_comparison(df, script_dir)
    
    plt.show()

def create_side_by_side_comparison(df, script_dir):
    """
    Versão com cada tipo de estação em subplot separado
    """
    ndbc_data = df[df['tipo_estacao'] == 'NDBC']
    emodnet_data = df[df['tipo_estacao'] == 'EMODNET']
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('WW3 Model Validation - General vs P99 Statistics by Station Type', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    metrics = [
        ('nbias_pct', 'nbias_p99_pct', 'NBIAS (%)', ['a', 'e']),
        ('nrmse_pct', 'nrmse_p99_pct', 'NRMSE (%)', ['b', 'f']),
        ('mae', 'mae_p99', 'MAE (m)', ['c', 'g']),
        ('si', 'si_p99', 'SI', ['d', 'h'])
    ]
    
    for col, (metric_gen, metric_p99, title, panel_labels) in enumerate(metrics):
        # NDBC (linha superior)
        ax_ndbc = axes[0, col]
        
        means_ndbc = [ndbc_data[metric_gen].mean(), ndbc_data[metric_p99].mean()]
        stds_ndbc = [ndbc_data[metric_gen].std(), ndbc_data[metric_p99].std()]
        
        bars = ax_ndbc.bar(['General', 'P99'], means_ndbc, yerr=stds_ndbc,
                          color=['#5B9BD5', '#2F5F8F'], capsize=5, alpha=0.8,
                          edgecolor='black', linewidth=0.5)
        
        ax_ndbc.set_title(f'NDBC - {title}', fontsize=12, fontweight='bold')
        ax_ndbc.grid(True, alpha=0.3, axis='y')
        
        if 'NBIAS' in title:
            ax_ndbc.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax_ndbc.text(0.02, 0.98, f'({panel_labels[0]})', transform=ax_ndbc.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
                    fontsize=12, fontweight='bold', verticalalignment='top')
        
        # Valores nas barras com posicionamento melhorado
        for bar, mean, std in zip(bars, means_ndbc, stds_ndbc):
            height = bar.get_height()
            if height < 0:
                y_pos = height - std - 0.3
                va = 'top'
            else:
                y_pos = height + std + 0.3
                va = 'bottom'
            ax_ndbc.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{mean:.1f}', ha='center', va=va, fontsize=10, fontweight='bold')
        
        # EMODNET (linha inferior)
        ax_emodnet = axes[1, col]
        
        means_emodnet = [emodnet_data[metric_gen].mean(), emodnet_data[metric_p99].mean()]
        stds_emodnet = [emodnet_data[metric_gen].std(), emodnet_data[metric_p99].std()]
        
        bars = ax_emodnet.bar(['General', 'P99'], means_emodnet, yerr=stds_emodnet,
                             color=['#A5A5A5', '#595959'], capsize=5, alpha=0.8,
                             edgecolor='black', linewidth=0.5)
        
        ax_emodnet.set_title(f'EMODNET - {title}', fontsize=12, fontweight='bold')
        ax_emodnet.grid(True, alpha=0.3, axis='y')
        
        if 'NBIAS' in title:
            ax_emodnet.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        
        ax_emodnet.text(0.02, 0.98, f'({panel_labels[1]})', transform=ax_emodnet.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
                       fontsize=12, fontweight='bold', verticalalignment='top')
        
        # Valores nas barras com posicionamento melhorado
        for bar, mean, std in zip(bars, means_emodnet, stds_emodnet):
            height = bar.get_height()
            if height < 0:
                y_pos = height - std - 0.3
                va = 'top'
            else:
                y_pos = height + std + 0.3
                va = 'bottom'
            ax_emodnet.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{mean:.1f}', ha='center', va=va, fontsize=10, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    
    output_file = os.path.join(script_dir, 'side_by_side_comparison_general_vs_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Figura lado a lado salva: {output_file}")

if __name__ == "__main__":
    create_simple_comparison()