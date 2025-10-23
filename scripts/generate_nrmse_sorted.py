#!/usr/bin/env python3
"""
Script para gerar lista ordenada por NRMSE (Normalized Root Mean Square Error)
"""

import pandas as pd
import os

def generate_nrmse_sorted_list():
    """
    Lê o arquivo de estatísticas e gera lista ordenada por NRMSE
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    output_file = os.path.join(script_dir, 'nrmse_sorted.txt')
    
    # Verificar se o arquivo de estatísticas existe
    if not os.path.exists(stats_file):
        print(f"Erro: Arquivo {stats_file} não encontrado!")
        return
    
    # Ler dados
    df = pd.read_csv(stats_file)
    
    # Calcular NRMSE (SI = RMSE/mean_obs)
    df['nrmse'] = df['si'] * 100  # Converter para percentual
    
    # Ordenar por NRMSE (menor para maior)
    df_sorted = df.sort_values('nrmse')
    
    # Gerar arquivo de saída
    with open(output_file, 'w') as f:
        f.write("Lista de estações ordenada por NRMSE (menor para maior):\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Estação':<10} {'NRMSE (%)':<12} {'Correlação':<12} {'N Pontos':<10}\n")
        f.write("-" * 60 + "\n")
        
        for _, row in df_sorted.iterrows():
            f.write(f"{row['estacao']:<10} {row['nrmse']:<12.2f} {row['correlacao']:<12.4f} {row['n_pontos']:<10.0f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"Total de estações: {len(df)}\n")
        f.write(f"NRMSE médio: {df['nrmse'].mean():.2f}%\n")
        f.write(f"Melhor estação (menor NRMSE): {df_sorted.iloc[0]['estacao']} ({df_sorted.iloc[0]['nrmse']:.2f}%)\n")
        f.write(f"Pior estação (maior NRMSE): {df_sorted.iloc[-1]['estacao']} ({df_sorted.iloc[-1]['nrmse']:.2f}%)\n")
    
    print(f"Lista ordenada por NRMSE salva em: {output_file}")
    print(f"Total de estações processadas: {len(df)}")
    print(f"NRMSE médio: {df['nrmse'].mean():.2f}%")
    
    # Mostrar top 10 melhores e 10 piores
    print("\nTOP 10 MELHORES ESTAÇÕES (menor NRMSE):")
    print("-" * 50)
    for i, (_, row) in enumerate(df_sorted.head(10).iterrows()):
        print(f"{i+1:2d}. {row['estacao']:<10} - NRMSE: {row['nrmse']:<6.2f}% (Corr: {row['correlacao']:.4f})")
    
    print("\nTOP 10 PIORES ESTAÇÕES (maior NRMSE):")
    print("-" * 50)
    for i, (_, row) in enumerate(df_sorted.tail(10).iterrows()):
        print(f"{i+1:2d}. {row['estacao']:<10} - NRMSE: {row['nrmse']:<6.2f}% (Corr: {row['correlacao']:.4f})")

if __name__ == "__main__":
    generate_nrmse_sorted_list()