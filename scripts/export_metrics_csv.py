#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta métricas de validação (NRMSE, NBIAS, Correlação) para CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Diretórios
SCRIPT_DIR = Path(__file__).parent
CSV_FILE = SCRIPT_DIR / "summary_statistics.csv"
STATIONS_FILE = SCRIPT_DIR.parent / "stations_match.csv"
OUTPUT_FILE = SCRIPT_DIR / "validation_metrics.csv"

def main():
    print("="*70)
    print("EXPORTANDO MÉTRICAS DE VALIDAÇÃO PARA CSV")
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
    
    # Seleciona apenas as colunas necessárias
    df_export = df_clean[['estacao', 'tipo', 'lon_obs', 'lat_obs', 
                           'nrmse', 'nbias', 'correlacao', 
                           'n_pontos', 'hs_obs_mean']].copy()
    
    # Renomeia colunas para inglês
    df_export.columns = ['station', 'type', 'longitude', 'latitude', 
                         'nrmse_pct', 'nbias_pct', 'correlation', 
                         'n_points', 'hs_obs_mean_m']
    
    # Ordena por tipo e depois por estação
    df_export = df_export.sort_values(['type', 'station'])
    
    # Salva CSV
    df_export.to_csv(OUTPUT_FILE, index=False, float_format='%.4f')
    
    print(f"\n✅ Arquivo salvo: {OUTPUT_FILE}")
    print(f"\n📊 Resumo:")
    print(f"   Total de estações: {len(df_export)}")
    print(f"   - NDBC: {len(df_export[df_export['type']=='NDBC'])}")
    print(f"   - EMODNET: {len(df_export[df_export['type']=='EMODNET'])}")
    
    # Mostra primeiras linhas
    print(f"\n📋 Primeiras 5 linhas:")
    print(df_export.head().to_string(index=False))
    
    print("\n" + "="*70)
    print("✅ EXPORTAÇÃO CONCLUÍDA!")
    print("="*70)
    print(f"\nColunas no arquivo CSV:")
    print(f"  - station: Nome/ID da estação")
    print(f"  - type: Tipo (NDBC ou EMODNET)")
    print(f"  - longitude: Longitude da estação")
    print(f"  - latitude: Latitude da estação")
    print(f"  - nrmse_pct: NRMSE normalizado em %")
    print(f"  - nbias_pct: NBIAS normalizado em %")
    print(f"  - correlation: Coeficiente de correlação")
    print(f"  - n_points: Número de pontos na série")
    print(f"  - hs_obs_mean_m: Hs médio observado em metros")
    print("="*70)

if __name__ == "__main__":
    main()
