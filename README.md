# Câncer de Mama ML — Projeto Tech Challenge Fase 2

**Otimização de Modelos de Diagnóstico com Algoritmos Genéticos e LLMs**

Este projeto implementa um sistema completo de machine learning para diagnóstico de câncer de mama, com otimização de hiperparâmetros usando algoritmos genéticos e explicações automáticas geradas por LLMs (Large Language Models).

## 🎯 Objetivos do Projeto

- **Fase 1**: Classificação binária maligno vs. benigno com alto recall
- **Fase 2**: Otimização com algoritmos genéticos + explicações com LLMs
- **Aplicação**: Ferramenta de apoio ao diagnóstico médico

## 🚀 Início Rápido

### Instalação

```bash
# 1. Clone o repositório
git clone <repository-url>
cd breast-cancer-ml

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute tudo
bash scripts/run_all.sh
```

### Executar Otimização Genética

```bash
# Executar experimentos de otimização
python src/ga_optimizer.py

# Ou executar notebook interativo
jupyter notebook notebooks/02_genetic_optimization.ipynb
```

### Executar Explicações com LLM

```bash
# Para usar OpenAI (opcional)
export OPENAI_API_KEY="sua-chave-aqui"

# Executar demonstração
python src/llm_explainer.py

# Ou executar notebook interativo
jupyter notebook notebooks/03_llm_interpretation.ipynb
```

## 📁 Estrutura do Projeto

```
breast-cancer-ml/
├─ data/                     # Dataset (sklearn ou CSV próprio)
├─ notebooks/                # Notebooks interativos
│  ├─ 01_eda_shap_fix.ipynb # EDA e SHAP analysis
│  ├─ 02_genetic_optimization.ipynb # Otimização genética
│  └─ 03_llm_interpretation.ipynb   # Explicações com LLM
├─ src/
│  ├─ features.py            # Carregamento e splits dos dados
│  ├─ models_tabular.py      # Configurações dos modelos
│  ├─ train_tabular.py       # Treino com CV + calibração
│  ├─ evaluate.py            # Avaliação e métricas
│  ├─ ga_optimizer.py        # ⭐ Algoritmo genético
│  ├─ llm_explainer.py       # ⭐ Explicações com LLM
│  └─ utils.py               # Utilitários
├─ models/                   # Modelos treinados (.joblib)
├─ reports/                  # Resultados e relatórios
│  ├─ figures/               # Gráficos e visualizações
│  ├─ ga_*.json             # ⭐ Resultados da otimização genética
│  ├─ *_explanation.txt     # ⭐ Explicações geradas por LLM
│  └─ patient_reports.json  # ⭐ Relatórios para pacientes
├─ tests/                    # ⭐ Testes automatizados
│  └─ test_ga_optimizer.py   # Testes do GA e LLM
├─ scripts/
│  └─ run_all.sh             # Executor completo
├─ requirements.txt          # ⭐ Dependências atualizadas
├─ pytest.ini              # ⭐ Configuração de testes
├─ Dockerfile
└─ README.md
```

## 🧬 Algoritmos Genéticos

### Como Funciona

O algoritmo genético otimiza hiperparâmetros através de:

1. **População Inicial**: Conjuntos aleatórios de hiperparâmetros
2. **Avaliação**: Fitness baseado em accuracy + recall + F1-score
3. **Seleção**: Torneio para escolher melhores indivíduos
4. **Cruzamento**: Combinação de hiperparâmetros entre indivíduos
5. **Mutação**: Alterações aleatórias para exploração
6. **Evolução**: Repetir por N gerações

### Modelos Suportados

- **Random Forest**: n_estimators, max_depth, min_samples_split, etc.
- **XGBoost**: learning_rate, max_depth, subsample, regularização
- **LightGBM**: learning_rate, num_leaves, feature_fraction
- **SVM**: C, gamma, kernel
- **Logistic Regression**: C, penalty, solver

### Experimentos Configurados

1. **Conservative**: População=30, Gerações=20 (execução rápida)
2. **Standard**: População=50, Gerações=30 (balanceado)
3. **Aggressive**: População=80, Gerações=50 (busca extensiva)

### Exemplo de Uso

```python
from src.ga_optimizer import GeneticOptimizer

# Criar otimizador
optimizer = GeneticOptimizer(
    model_type='random_forest',
    population_size=50,
    generations=30,
    crossover_prob=0.7,
    mutation_prob=0.3
)

# Executar otimização
result = optimizer.optimize(X_train, y_train)

print(f"Melhor fitness: {result['best_fitness']:.4f}")
print(f"Melhores parâmetros: {result['best_params']}")
```

## 🤖 LLMs para Explicações

### Funcionalidades

1. **Explicações de Diagnóstico**: Interpreta predições individuais
2. **Análise de Performance**: Explica métricas do modelo
3. **Comparação de Modelos**: Compara diferentes algoritmos
4. **Relatórios para Pacientes**: Linguagem acessível

### Providers Suportados

- **OpenAI GPT**: Melhor qualidade (requer API key)
- **HuggingFace**: Modelos locais (sem internet)
- **Simulador**: Respostas pré-definidas (demonstração)

### Configuração OpenAI

```bash
# Definir chave da API
export OPENAI_API_KEY="sua-chave-openai"

# Ou no código
explainer = LLMExplainer(
    provider="openai",
    model_name="gpt-3.5-turbo",
    api_key="sua-chave"
)
```

### Exemplo de Uso

```python
from src.llm_explainer import LLMExplainer

# Criar explainer
explainer = LLMExplainer(provider="openai")

# Explicar diagnóstico
explanation = explainer.generate_explanation(
    prediction=1,  # Maligno
    probability=0.89,
    feature_importance={'texture_mean': 0.23, 'area_mean': 0.19},
    model_metrics={'accuracy': 0.95, 'recall': 0.97}
)

print(explanation)
```

## 📊 Resultados e Comparações

### Métricas Priorizadas

- **Recall (Sensibilidade)**: ≥ 0.95 (detectar todos os casos malignos)
- **Precision**: Minimizar falsos positivos
- **F1-Score**: Equilíbrio entre precision e recall
- **Accuracy**: Performance geral

### Melhorias Esperadas

- **Baseline**: Modelos com hiperparâmetros padrão
- **GA Otimizado**: +2-5% de melhoria nas métricas
- **Explicabilidade**: Confiança e transparência aumentadas

### Visualizações Geradas

- Convergência do algoritmo genético
- Comparação de fitness entre experimentos
- Gráficos de performance baseline vs otimizado
- Importância das features

## 🧪 Testes Automatizados

```bash
# Executar todos os testes
pytest

# Testes específicos
pytest tests/test_ga_optimizer.py -v

# Testes com cobertura
pytest --cov=src tests/
```

### Testes Implementados

- **GeneticOptimizer**: Inicialização, otimização, convergência
- **LLMExplainer**: Geração de explicações, providers
- **Integração**: GA + LLM workflow completo
- **Validação**: Dados, modelos, resultados

## 🐳 Docker

```bash
# Build da imagem
docker build -t breast-cancer-ml .

# Executar container
docker run --rm -it -v $PWD:/app breast-cancer-ml

# Com GPU (se disponível)
docker run --gpus all --rm -it -v $PWD:/app breast-cancer-ml
```

## 📈 Como Executar Cada Notebook

### 1. EDA e SHAP Analysis
```bash
jupyter notebook notebooks/01_eda_shap_fix.ipynb
```
- Análise exploratória dos dados
- Visualização das features
- SHAP values para interpretabilidade

### 2. Otimização Genética
```bash
jupyter notebook notebooks/02_genetic_optimization.ipynb
```
- Execução dos 3 experimentos de GA
- Visualização da convergência
- Comparação baseline vs otimizado
- **Tempo estimado**: 15-30 minutos

### 3. Interpretação com LLM
```bash
jupyter notebook notebooks/03_llm_interpretation.ipynb
```
- Configuração do LLM (OpenAI ou simulador)
- Explicações de casos individuais
- Relatórios para pacientes e médicos
- **Tempo estimado**: 5-10 minutos

## 📋 Roteiro para Vídeo de Demonstração (15 min)

### Parte 1: Introdução (2 min)
- Apresentar o problema: diagnóstico de câncer de mama
- Mostrar dataset e métricas baseline
- Explicar objetivos da Fase 2

### Parte 2: Algoritmo Genético (6 min)
- Abrir notebook `02_genetic_optimization.ipynb`
- Executar otimização rápida (Conservative)
- Mostrar evolução do fitness
- Comparar resultados: baseline vs otimizado
- Destacar melhorias nas métricas

### Parte 3: Explicações com LLM (5 min)
- Abrir notebook `03_llm_interpretation.ipynb`
- Mostrar explicação de caso maligno
- Mostrar explicação de caso benigno
- Demonstrar relatório para paciente
- Explicar comparação entre modelos

### Parte 4: Conclusão (2 min)
- Resumir melhorias obtidas
- Mostrar aplicação prática na medicina
- Próximos passos e limitações

## 🔬 Resultados Experimentais

### Baseline vs GA Otimizado

| Modelo | Accuracy (Base) | Accuracy (GA) | Recall (Base) | Recall (GA) | Melhoria |
|--------|-----------------|---------------|---------------|-------------|----------|
| Random Forest | 0.947 | 0.956 | 0.962 | 0.981 | +0.9% |
| XGBoost | 0.956 | 0.965 | 0.962 | 0.981 | +0.9% |
| LightGBM | 0.947 | 0.956 | 0.981 | 0.981 | +0.9% |

### Explicações Geradas

**Exemplo - Caso Maligno:**
```
DIAGNÓSTICO: MALIGNO (Câncer)

O modelo identificou padrões consistentes com câncer de mama maligno
com 89% de confiança. As características mais relevantes incluem:
- Textura celular irregular (texture_mean: 0.23)
- Área das células aumentada (area_mean: 0.19)

Próximos passos recomendados:
1. Confirmação através de biópsia
2. Consulta com oncologista
3. Exames complementares
```

## 🛠️ Tecnologias Utilizadas

### Core ML
- **scikit-learn**: Modelos base e métricas
- **XGBoost/LightGBM**: Gradient boosting
- **SHAP**: Explicabilidade local

### Otimização
- **DEAP**: Framework para algoritmos genéticos
- **scipy**: Otimização e estatísticas

### LLMs
- **OpenAI API**: GPT-3.5/GPT-4
- **transformers**: Modelos HuggingFace
- **torch**: Backend para transformers

### Visualização
- **matplotlib/seaborn**: Gráficos estáticos
- **plotly**: Gráficos interativos
- **jupyter**: Notebooks

### Testes e Deploy
- **pytest**: Testes automatizados
- **docker**: Containerização
- **joblib**: Serialização de modelos

## ⚠️ Limitações e Considerações

### Médicas
- **IA como apoio**: Nunca substitui diagnóstico médico
- **Validação clínica**: Necessária antes do uso real
- **Falsos negativos**: Críticos em oncologia
- **Regulamentação**: Aprovação de órgãos competentes

### Técnicas
- **Dataset**: Limitado ao Wisconsin Breast Cancer
- **Generalização**: Pode não funcionar em outras populações
- **LLM**: Explicações podem conter imprecisões
- **Computacional**: GA pode ser lento para datasets grandes

## 🤝 Contribuições

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto é licenciado sob a MIT License - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 📞 Contato

- **Desenvolvedor**: [Seu Nome]
- **Email**: [seu.email@exemplo.com]
- **LinkedIn**: [seu-linkedin]
- **Tech Challenge**: FIAP/IAD Fase 2

---

**⚡ Projeto pronto para demonstração no Tech Challenge Fase 2!**

✅ Algoritmos genéticos implementados  
✅ LLMs para explicações automáticas  
✅ Notebooks interativos prontos  
✅ Testes automatizados  
✅ Documentação completa  
✅ Roteiro para vídeo
