#!/usr/bin/env python3
"""
Script para analisar e visualizar o fenômeno de correlação negativa no P99
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def analyze_p99_correlation_paradox():
    """
    Analisa o paradoxo da correlação negativa no P99 vs correlação alta total
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ler dados das estações com correlação negativa P99
    negative_p99_file = os.path.join(script_dir, 'estacoes_correlacao_negativa_p99.csv')
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    
    if not os.path.exists(negative_p99_file):
        print("Arquivo estacoes_correlacao_negativa_p99.csv não encontrado!")
        return
    
    df_negative = pd.read_csv(negative_p99_file)
    df_all = pd.read_csv(stats_file)
    
    print("🔍 ANÁLISE DO PARADOXO: CORRELAÇÃO ALTA vs P99 NEGATIVA")
    print("="*70)
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Análise do Paradoxo: Correlação Total vs P99', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot: Correlação Total vs Correlação P99
    ax1 = axes[0, 0]
    
    # Todas as estações
    scatter = ax1.scatter(df_all['correlacao'], df_all['corr_p99'], 
                         alpha=0.6, c='lightblue', s=30, label='Todas as estações')
    
    # Estações com P99 negativa
    ax1.scatter(df_negative['correlacao'], df_negative['corr_p99'], 
               c='red', s=60, alpha=0.8, label='P99 Negativa', marker='x')
    
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Corr > 0.9')
    
    ax1.set_xlabel('Correlação Total')
    ax1.set_ylabel('Correlação P99')
    ax1.set_title('Correlação Total vs P99')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histograma das diferenças
    ax2 = axes[0, 1]
    
    # Calcular diferença entre P99 observado e modelado para estações negativas
    df_negative['p99_diff'] = df_negative['p99_mod'] - df_negative['p99_obs']
    df_negative['p99_ratio'] = df_negative['p99_mod'] / df_negative['p99_obs']
    
    ax2.hist(df_negative['p99_ratio'], bins=10, alpha=0.7, color='red', edgecolor='black')
    ax2.axvline(x=1.0, color='black', linestyle='--', label='P99_mod = P99_obs')
    ax2.set_xlabel('Razão P99 (Modelo/Observado)')
    ax2.set_ylabel('Frequência')
    ax2.set_title('Distribuição P99: Modelo vs Observado')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. NRMSE Total vs NRMSE P99
    ax3 = axes[1, 0]
    
    ax3.scatter(df_negative['nrmse_pct'], df_negative['nrmse_p99_pct'], 
               c='red', s=60, alpha=0.8)
    
    # Linha 1:1
    max_nrmse = max(df_negative['nrmse_pct'].max(), df_negative['nrmse_p99_pct'].max())
    ax3.plot([0, max_nrmse], [0, max_nrmse], 'k--', alpha=0.5, label='1:1')
    
    ax3.set_xlabel('NRMSE Total (%)')
    ax3.set_ylabel('NRMSE P99 (%)')
    ax3.set_title('NRMSE: Total vs P99')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Adicionar labels das estações
    for _, row in df_negative.iterrows():
        ax3.annotate(str(int(row['estacao'])), 
                    (row['nrmse_pct'], row['nrmse_p99_pct']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    # 4. Análise por tipo de estação
    ax4 = axes[1, 1]
    
    tipo_counts = df_negative['tipo_estacao'].value_counts()
    colors = ['lightcoral', 'lightskyblue']
    ax4.pie(tipo_counts.values, labels=tipo_counts.index, autopct='%1.1f%%',
           colors=colors, startangle=90)
    ax4.set_title('Distribuição por Tipo de Estação\n(P99 Negativa)')
    
    plt.tight_layout()
    
    # Salvar figura
    output_file = os.path.join(script_dir, 'analise_paradoxo_p99.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Figura salva: {output_file}")
    
    # Análise estatística detalhada
    print("\n📊 ANÁLISE ESTATÍSTICA DETALHADA:")
    print("="*70)
    
    print("🔴 ESTAÇÕES COM CORRELAÇÃO P99 NEGATIVA:")
    print(f"   Total: {len(df_negative)}")
    print(f"   NDBC: {len(df_negative[df_negative['tipo_estacao'] == 'NDBC'])}")
    print(f"   EMODNET: {len(df_negative[df_negative['tipo_estacao'] == 'EMODNET'])}")
    
    print(f"\n📈 ESTATÍSTICAS DESCRITIVAS:")
    print(f"   Correlação Total média: {df_negative['correlacao'].mean():.4f}")
    print(f"   Correlação P99 média: {df_negative['corr_p99'].mean():.4f}")
    print(f"   NRMSE Total médio: {df_negative['nrmse_pct'].mean():.2f}%")
    print(f"   NRMSE P99 médio: {df_negative['nrmse_p99_pct'].mean():.2f}%")
    
    print(f"\n🌊 ANÁLISE DOS EVENTOS EXTREMOS:")
    print(f"   P99 Observado médio: {df_negative['p99_obs'].mean():.2f} m")
    print(f"   P99 Modelado médio: {df_negative['p99_mod'].mean():.2f} m")
    print(f"   Razão média (Mod/Obs): {df_negative['p99_ratio'].mean():.3f}")
    
    # Identificar padrões
    underestimate = df_negative[df_negative['p99_ratio'] < 1.0]
    overestimate = df_negative[df_negative['p99_ratio'] > 1.0]
    
    print(f"\n⬇️  MODELO SUBESTIMA P99: {len(underestimate)} estações")
    print(f"⬆️  MODELO SUPERESTIMA P99: {len(overestimate)} estações")
    
    # Casos mais extremos
    print(f"\n🔴 CASOS MAIS EXTREMOS (Corr P99 < -0.3):")
    extreme_cases = df_negative[df_negative['corr_p99'] < -0.3]
    for _, row in extreme_cases.iterrows():
        regime = "subestima" if row['p99_ratio'] < 1.0 else "superestima"
        print(f"   {int(row['estacao'])} ({row['tipo_estacao']}): "
              f"Corr={row['correlacao']:.3f} | P99_corr={row['corr_p99']:.3f} | "
              f"Modelo {regime} ({row['p99_ratio']:.2f}x)")
    
    print(f"\n💡 POSSÍVEIS EXPLICAÇÕES:")
    print("="*70)
    print("1. 🌀 EVENTOS EXTREMOS têm física diferente das ondas normais")
    print("2. 🏔️  TOPOGRAFIA local pode afetar ondas grandes de forma diferente")
    print("3. 🌊 SHOALING e BREAKING podem não estar bem representados")
    print("4. 📊 AMOSTRAGEM limitada de eventos extremos (poucos dados P99)")
    print("5. 🌪️  TEMPESTADES e ciclones podem ter padrões específicos não capturados")
    print("6. 📍 RESOLUÇÃO do modelo pode ser inadequada para eventos extremos locais")

if __name__ == "__main__":
    analyze_p99_correlation_paradox()