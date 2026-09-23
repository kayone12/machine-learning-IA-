# Machine Learning - IA
Atividades práticas de Machine Learning em Python (ADS/UNIT).

## Estrutura
- `tratamento-de-dados/` - limpeza e preparação de base de funcionários da construção (30 registros). Tratamento de nulos com média, normalização Min-Max, label encoding em `desempenho` e one-hot em `cargo`/`setor` com pandas/numpy.
- `knn-naive-bayes/` - comparação KNN (K=1 a 10) vs Naive Bayes Gaussian no dataset Iris, com KFold 5 e acurácia média.
- `knn-minha-base/` - pipeline completo na base `cardekho_dataset.csv` (venda de carros). Mesmo tratamento + classificação de `seller_type` com KNN vs Naive Bayes.

## Como rodar
```bash
pip install -r requirements.txt
python pasta-do-exemplo/arquivo.py
