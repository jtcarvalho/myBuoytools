#!/usr/bin/env python3
"""
Script para exportar todas as estatísticas do terminal para CSV e identificar estações com correlação negativa no P99
"""

import pandas as pd
import numpy as np
import os

def export_all_statistics_to_csv():
    """
    Exporta todas as estatísticas calculadas para arquivos CSV
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stats_file = os.path.join(script_dir, 'summary_statistics.csv')
    
    if not os.path.exists(stats_file):
        print(f"Erro: Arquivo {stats_file} não encontrado!")
        return
    
    # Ler dados
    df = pd.read_csv(stats_file)
    print(f"📊 Dados carregados: {len(df)} estações")
    
    # Calcular NRMSE (SI = RMSE/mean_obs)
    df['nrmse_pct'] = df['si'] * 100
    df['nbias_pct'] = (df['bias'] / df['hs_obs_mean']) * 100
    
    # Classificar por tipo de estação
    df['tipo_estacao'] = df['estacao'].apply(lambda x: 'NDBC' if str(x).startswith(('4', '5')) else 'EMODNET')
    
    # ============================================
    # 1. ESTATÍSTICAS GERAIS POR TIPO DE ESTAÇÃO
    # ============================================
    print("\n🔄 Gerando estatísticas gerais por tipo de estação...")
    
    stats_gerais = []
    for tipo in ['NDBC', 'EMODNET']:
        subset = df[df['tipo_estacao'] == tipo]
        if len(subset) > 0:
            stats_gerais.append({
                'tipo_estacao': tipo,
                'n_estacoes': len(subset),
                'nbias_mean_pct': subset['nbias_pct'].mean(),
                'nbias_std_pct': subset['nbias_pct'].std(),
                'nrmse_mean_pct': subset['nrmse_pct'].mean(),
                'nrmse_std_pct': subset['nrmse_pct'].std(),
                'correlacao_mean': subset['correlacao'].mean(),
                'correlacao_std': subset['correlacao'].std(),
                'mae_mean_m': subset['mae'].mean(),
                'mae_std_m': subset['mae'].std(),
                'si_mean': subset['si'].mean(),
                'si_std': subset['si'].std(),
                'index_agreement_mean': subset['index_agreement'].mean(),
                'index_agreement_std': subset['index_agreement'].std(),
                'bias_mean_m': subset['bias'].mean(),
                'rmse_mean_m': subset['rmse'].mean(),
                'hs_obs_mean_m': subset['hs_obs_mean'].mean()
            })
    
    df_stats_gerais = pd.DataFrame(stats_gerais)
    df_stats_gerais.to_csv(os.path.join(script_dir, 'estatisticas_gerais_por_tipo.csv'), index=False)
    print("✅ Salvo: estatisticas_gerais_por_tipo.csv")
    
    # ============================================
    # 2. ESTATÍSTICAS P99 POR TIPO DE ESTAÇÃO
    # ============================================
    print("\n🔄 Gerando estatísticas P99 por tipo de estação...")
    
    # Calcular estatísticas P99
    df['nbias_p99_pct'] = (df['bias_p99'] / df['p99_obs']) * 100
    df['nrmse_p99_pct'] = (df['rmse_p99'] / df['p99_obs']) * 100
    df['si_p99'] = df['rmse_p99'] / df['p99_obs']
    
    stats_p99 = []
    for tipo in ['NDBC', 'EMODNET']:
        subset = df[df['tipo_estacao'] == tipo]
        if len(subset) > 0:
            stats_p99.append({
                'tipo_estacao': tipo,
                'n_estacoes': len(subset),
                'p99_obs_mean_m': subset['p99_obs'].mean(),
                'p99_obs_std_m': subset['p99_obs'].std(),
                'nbias_p99_mean_pct': subset['nbias_p99_pct'].mean(),
                'nbias_p99_std_pct': subset['nbias_p99_pct'].std(),
                'nrmse_p99_mean_pct': subset['nrmse_p99_pct'].mean(),
                'nrmse_p99_std_pct': subset['nrmse_p99_pct'].std(),
                'corr_p99_mean': subset['corr_p99'].mean(),
                'corr_p99_std': subset['corr_p99'].std(),
                'mae_p99_mean_m': subset['mae_p99'].mean(),
                'mae_p99_std_m': subset['mae_p99'].std(),
                'si_p99_mean': subset['si_p99'].mean(),
                'si_p99_std': subset['si_p99'].std(),
                'index_agreement_mean': subset['index_agreement'].mean(),
                'index_agreement_std': subset['index_agreement'].std()
            })
    
    df_stats_p99 = pd.DataFrame(stats_p99)
    df_stats_p99.to_csv(os.path.join(script_dir, 'estatisticas_p99_por_tipo.csv'), index=False)
    print("✅ Salvo: estatisticas_p99_por_tipo.csv")
    
    # ============================================
    # 3. ESTATÍSTICAS POR REGIÃO (simulado)
    # ============================================
    print("\n🔄 Gerando estatísticas por região...")
    
    def classify_region(estacao):
        """Classificar região baseado no ID da estação"""
        estacao_str = str(estacao)
        if estacao_str.startswith('4'):
            return 'North Atlantic'
        elif estacao_str.startswith('5'):
            return 'North Pacific'  
        elif estacao_str.startswith('66'):
            return 'Tropical Atlantic'
        elif estacao_str.startswith('55'):
            return 'Southern Ocean'
        else:
            return 'Other'
    
    df['regiao'] = df['estacao'].apply(classify_region)
    
    stats_regiao = []
    for regiao in df['regiao'].unique():
        subset = df[df['regiao'] == regiao]
        if len(subset) > 0:
            stats_regiao.append({
                'regiao': regiao,
                'n_estacoes': len(subset),
                'nrmse_mean_pct': subset['nrmse_pct'].mean(),
                'nbias_mean_pct': subset['nbias_pct'].mean(),
                'correlacao_mean': subset['correlacao'].mean()
            })
    
    df_stats_regiao = pd.DataFrame(stats_regiao)
    df_stats_regiao.to_csv(os.path.join(script_dir, 'estatisticas_por_regiao.csv'), index=False)
    print("✅ Salvo: estatisticas_por_regiao.csv")
    
    # ============================================
    # 4. ESTAÇÕES COM CORRELAÇÃO NEGATIVA NO P99
    # ============================================
    print("\n🔍 Identificando estações com correlação negativa no P99...")
    
    # Filtrar estações com correlação P99 negativa
    negative_corr_p99 = df[df['corr_p99'] < 0].copy()
    
    if len(negative_corr_p99) > 0:
        # Ordenar por correlação P99 (da mais negativa para menos negativa)
        negative_corr_p99 = negative_corr_p99.sort_values('corr_p99')
        
        # Selecionar colunas relevantes
        negative_corr_export = negative_corr_p99[['estacao', 'tipo_estacao', 'corr_p99', 'correlacao', 
                                                  'nrmse_p99_pct', 'nrmse_pct', 'p99_obs', 'p99_mod', 
                                                  'n_pontos', 'hs_obs_mean']].copy()
        
        negative_corr_export.to_csv(os.path.join(script_dir, 'estacoes_correlacao_negativa_p99.csv'), index=False)
        print(f"✅ Salvo: estacoes_correlacao_negativa_p99.csv ({len(negative_corr_p99)} estações)")
        
        print(f"\n❌ ESTAÇÕES COM CORRELAÇÃO NEGATIVA NO P99: {len(negative_corr_p99)}")
        print("="*80)
        for _, row in negative_corr_p99.iterrows():
            print(f"🔴 {row['estacao']:<10} ({row['tipo_estacao']:<7}) | Corr P99: {row['corr_p99']:<7.4f} | Corr Geral: {row['correlacao']:<7.4f} | NRMSE P99: {row['nrmse_p99_pct']:<7.2f}%")
    else:
        print("✅ Nenhuma estação com correlação negativa no P99 encontrada!")
    
    # ============================================
    # 5. TOP/BOTTOM RANKINGS
    # ============================================
    print("\n🔄 Gerando rankings...")
    
    # Top 10 melhores NRMSE
    top_nrmse = df.nsmallest(10, 'nrmse_pct')[['estacao', 'tipo_estacao', 'nrmse_pct', 'correlacao', 'n_pontos']].copy()
    top_nrmse.to_csv(os.path.join(script_dir, 'top10_melhores_nrmse.csv'), index=False)
    print("✅ Salvo: top10_melhores_nrmse.csv")
    
    # Top 10 piores NRMSE
    bottom_nrmse = df.nlargest(10, 'nrmse_pct')[['estacao', 'tipo_estacao', 'nrmse_pct', 'correlacao', 'n_pontos']].copy()
    bottom_nrmse.to_csv(os.path.join(script_dir, 'top10_piores_nrmse.csv'), index=False)
    print("✅ Salvo: top10_piores_nrmse.csv")
    
    # ============================================
    # 6. RESUMO FINAL
    # ============================================
    print("\n📊 RESUMO FINAL:")
    print("="*60)
    print(f"Total de estações processadas: {len(df)}")
    print(f"NRMSE médio: {df['nrmse_pct'].mean():.2f}%")
    print(f"Correlação média: {df['correlacao'].mean():.4f}")
    print(f"NBIAS médio: {df['nbias_pct'].mean():.2f}%")
    print(f"Estações NDBC: {len(df[df['tipo_estacao'] == 'NDBC'])}")
    print(f"Estações EMODNET: {len(df[df['tipo_estacao'] == 'EMODNET'])}")
    print(f"Estações com correlação P99 negativa: {len(negative_corr_p99)}")
    
    print("\n📁 ARQUIVOS CSV GERADOS:")
    print("="*60)
    csv_files = [
        'estatisticas_gerais_por_tipo.csv',
        'estatisticas_p99_por_tipo.csv', 
        'estatisticas_por_regiao.csv',
        'top10_melhores_nrmse.csv',
        'top10_piores_nrmse.csv'
    ]
    if len(negative_corr_p99) > 0:
        csv_files.append('estacoes_correlacao_negativa_p99.csv')
    
    for csv_file in csv_files:
        print(f"  • {csv_file}")

if __name__ == "__main__":
    export_all_statistics_to_csv()