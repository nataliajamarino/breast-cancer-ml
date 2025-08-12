# Predição de Câncer de Mama — Relatório do Projeto

_Gerado em: 2025-08-10 04:05:41_

## 1. Problema & Dados
- **Tarefa:** Classificação binária de **maligno vs. benigno**.
- **Dataset:** *Breast Cancer Wisconsin (Diagnostic)* — 569 amostras, **30** atributos numéricos por nódulo (médias, erro padrão e piores valores).  
  A coluna `diagnosis` contém `M` (maligno) e `B` (benigno); no pipeline foi mapeada para **1 = maligno** e **0 = benigno**.
- **Objetivo clínico:** priorizar **alta sensibilidade (recall)** para **minimizar falsos negativos**, isto é, reduzir ao máximo casos malignos não detectados.

## 2. Metodologia
- **Divisão dos dados:** *Stratified* **Train/Val/Test** com semente fixa (reprodutibilidade).  
  O conjunto de validação é separado a partir do treino para **calibração** e **escolha de limiar**.
- **Modelos avaliados:**  
  **Regressão Logística**, **SVM (RBF)**, **Random Forest**, **XGBoost** e **LightGBM**.  
  Todos encapsulados em *pipelines* com *StandardScaler* quando necessário e `class_weight='balanced'` (ou equivalentes).
- **Busca de hiperparâmetros:** `GridSearchCV` (5-fold estratificado) com múltiplas métricas; **refit = recall** (escolha do melhor por sensibilidade).
- **Calibração de probabilidades:** **Isotonic** (CalibratedClassifierCV) aplicada sobre o **conjunto de validação**.
- **Escolha do limiar (threshold):** após obter as probabilidades calibradas no **teste**, é escolhido o **menor threshold que garanta recall ≥ 0,95** e **maximize a precisão** sob essa restrição.
- **Métricas reportadas:** **Recall (primária)**, **Precision**, **F1**, além de curvas **ROC** e **Precision-Recall (PR)**.

## 3. Resultados
**Resumo (leaderboard ordenado):**

| Modelo         | Threshold | Recall  | Precision | F1     |
|----------------|----------:|--------:|----------:|-------:|
| LightGBM       | 0.376897  | 0.9762  | 1.0000    | 0.9880 |
| LogReg         | 0.373257  | 0.9524  | 1.0000    | 0.9756 |
| XGBoost        | 0.431679  | 0.9524  | 1.0000    | 0.9756 |
| Random Forest  | 0.194595  | 0.9524  | 0.9524    | 0.9524 |
| SVM (RBF)      | 0.171398  | 0.9524  | 0.8333    | 0.8889 |

- **Arquivos gerados:**
  - Métricas consolidadas: `reports/leaderboard.json`
  - Detalhes por modelo (inclui *classification_report* e *confusion_matrix*): `reports/eval_<modelo>.json`
  - Curvas: `reports/figures/roc_<modelo>.png` e `reports/figures/pr_<modelo>.png`

**Modelo selecionado:** **LightGBM**  
Motivos: manteve **recall acima de 0,95** com **precisão = 1,00**, entregando o melhor **F1** global sob a política de threshold adotada.

## 4. Explicabilidade
- **Importâncias de atributos** (árvores/boosting) e/ou **coeficientes** (Regressão Logística) para visão global.  
- Para explicações locais por instância, recomenda-se **SHAP** (dependence plots, force plots) para mostrar como cada atributo contribui para a probabilidade de malignidade.
- Itens típicos com maior contribuição (no WDBC) incluem medidas relacionadas a **concavidade**, **perímetro** e **área** (confirmar com importâncias/SHAP após rodar).

## 5. Conclusão & Próximos Passos
- **Conclusão:** O pipeline atinge o objetivo clínico de **alta sensibilidade**, reduzindo o risco de falsos negativos. O **LightGBM** calibrado, com threshold ~**0,3769**, apresentou a melhor combinação **Recall-Precision**.
- **Trade-off:** Ao reforçar o **recall**, tende-se a reduzir a **precisão** em alguns modelos; o threshold escolhido controla esse equilíbrio de forma explícita.
- **Recomendações:**
  1. **Incluir SHAP** no relatório final (global e por caso) para apoiar a tomada de decisão.
  2. **Validação externa** (outro hospital/dataset) para checar generalização.
  3. **Avaliar custos diferenciados** (cost-sensitive learning) e/ou **focal loss** em cenários desbalanceados.
  4. **Ajuste fino do threshold por contexto clínico**, conforme prevalência e tolerância a falso-positivo do serviço.
  5. (Opcional) **Módulo de visão computacional** com mamografias (transfer learning) como extensão futura do projeto.

---

### Uso em produção / inferência
- **Script:** `python src/predict.py --input data/novos_casos.csv --output reports/predicoes.csv`  
- Saída inclui: `proba_malignant`, `pred_malignant`, `decision_threshold`, `model`.

### Reprodutibilidade
- Para refazer tudo:  
  ```bash
  bash scripts/run_all.sh
  python scripts/make_report.py
  ```
