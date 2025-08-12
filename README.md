# Câncer de Mama ML — Projeto de Ponta a Ponta

Classificação binária **maligno vs. benigno** com foco em **alto recall (sensibilidade)**.  
A pipeline tabular está completa e reprodutível; o módulo opcional de visão está esboçado.

## Início rápido (Local)

```bash
# (1) Crie um ambiente virtual (opcional, mas recomendado)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# (2) Instale as dependências
pip install -r requirements.txt

# (3) Execute tudo
bash scripts/run_all.sh
```

Os artefatos vão para:
- `models/` — modelos calibrados (`*.joblib`) e `registry.json`
- `reports/` — `leaderboard.json`, `eval_*.json` e `figures/` com curvas ROC/PR

## Docker

```bash
docker build -t breast-cancer-ml .
docker run --rm -it -v $PWD:/app breast-cancer-ml
```

## Estrutura do Projeto

```
breast-cancer-ml/
├─ data/                     # carregado automaticamente do sklearn; aceita CSVs próprios
├─ notebooks/                # notebooks de EDA (opcional)
├─ src/
│  ├─ features.py            # carregamento do dataset & splits
│  ├─ models_tabular.py      # pipelines + grades de busca
│  ├─ train_tabular.py       # treino com CV + calibração isotônica + registro
│  ├─ evaluate.py            # avaliação no teste + curvas + escolha de threshold
│  └─ utils.py               # utilitários para busca de limiar (threshold)
├─ vision/                   # bônus opcional (transfer learning)
├─ reports/
│  ├─ figures/               # gráficos salvos
│  └─ report_template.md     # modelo para escrever o relatório
├─ scripts/
│  └─ run_all.sh             # executor ponta a ponta
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

## Por que esses modelos?
- **SVM (RBF), XGBoost, LightGBM** costumam ir muito bem em dados tabulares.
- **Regressão Logística** fornece um baseline simples e bem calibrado.
- **Random Forest** é robusto e suficientemente interpretável para triagem de features.

## Métricas & Limiar (Threshold)
Otimizamos para **recall ≥ 0,95** e então escolhemos a melhor **precisão** sob essa restrição.  
Também produzimos **ROC-AUC** e **PR-AUC**, e salvamos relatórios **JSON por modelo** para rastreabilidade.

## Observações
- Por padrão, remapeamos os rótulos para **1 = maligno**, **0 = benigno** para alinhamento clínico.
- A calibração usa **isotônica** no conjunto de validação para melhorar a qualidade das probabilidades.

---

## Usando seu próprio CSV (`data/data.csv`)
Coloque um arquivo chamado `data.csv` na pasta `data/`. A pipeline vai **preferi-lo** ao dataset embutido.  
Ele deve incluir a coluna `diagnosis` (`'M'/'B'` ou `1/0`). As colunas `id` e `Unnamed: 32` (se existirem) são descartadas.
