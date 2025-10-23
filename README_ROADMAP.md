# Roadmap de Execução para Gerar `validation_comparison.png`

Este roadmap mostra a ordem recomendada de execução dos scripts, os arquivos gerados em cada etapa e dependências para obter a figura final de comparação de validação.

## 1. Preparação dos Dados
- **Verifique se os dados das boias (NDBC, EMODNET) e do modelo WW3 estão nas pastas corretas:**
  - `~/Downloads/ndbc` (ou caminho customizado)
  - `~/Downloads/emodnet` (ou caminho customizado)
- **Certifique-se de que o arquivo `stations_match.csv` está presente na raiz do projeto ou pasta scripts.**

## 2. Checagem dos Dados
- **Script:** `scripts/check-dados.py`
- **Função:** Verifica e organiza os dados das estações.
- **Arquivo gerado:** (nenhum obrigatório, mas pode atualizar `stations_match.csv`)

## 3. Estatísticas Resumidas
- **Script:** `scripts/create_summary_stats.py`
- **Função:** Calcula estatísticas gerais para cada estação (boia x modelo).
- **Arquivo gerado:** `scripts/summary_statistics.csv`

## 4. Exportação de Métricas Compactas
- **Script:** `scripts/export_metrics_csv.py`
- **Função:** Exporta métricas compactas e coordenadas das estações.
- **Arquivo gerado:** `scripts/validation_metrics.csv`

## 5. (Opcional) Outras Estatísticas e Mapas
- Scripts como `create_clean_distributions.py`, `create_p99_maps.py`, `create_comparative_maps.py` podem ser usados para análises adicionais, mas não são obrigatórios para a figura final.

## 6. Geração da Figura Comparativa
- **Script:** `scripts/create_validation_comparison.py`
- **Função:** Gera a figura comparativa dos mapas de validação geral e P99.
- **Arquivos necessários:**
  - `summary_statistics.csv`
  - `validation_metrics.csv`
  - `stations_match.csv`
- **Arquivo gerado:** `scripts/validation_comparison.png`

---

## Fluxo Resumido
1. Organize os dados e `stations_match.csv`
2. Execute `check-dados.py` (opcional)
3. Execute `create_summary_stats.py`
4. Execute `export_metrics_csv.py`
5. Execute `create_validation_comparison.py`

Ao final, o arquivo `validation_comparison.png` será gerado na pasta `scripts/`.

---

## Observações
- Se mudar os caminhos dos dados, use os argumentos dos scripts para ajustar.
- Para análises extras, consulte os outros scripts na pasta `scripts/`.
