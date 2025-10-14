#!/bin/bash

# Script para executar o pipeline completo do Tech Challenge Fase 2
# Inclui otimização genética e explicações com LLM

echo "🚀 Executando Tech Challenge Fase 2 - Pipeline Completo"
echo "======================================================"

# Verificar se estamos no diretório correto
if [ ! -f "requirements.txt" ]; then
    echo "❌ Execute este script a partir do diretório raiz do projeto"
    exit 1
fi

# Instalar dependências se necessário
echo "📦 Verificando dependências..."
pip install -r requirements.txt > /dev/null 2>&1

# Executar pipeline original (Fase 1)
echo "🔄 Executando pipeline original (Fase 1)..."
bash scripts/run_all.sh

# Executar otimização genética
echo ""
echo "🧬 Iniciando otimização genética..."
echo "⏰ Isso pode levar 10-20 minutos..."
python src/ga_optimizer.py

# Executar explicações com LLM
echo ""
echo "🤖 Gerando explicações com LLM..."
python src/llm_explainer.py

# Executar testes
echo ""
echo "🧪 Executando testes automatizados..."
pytest tests/ -v --tb=short

# Relatório final
echo ""
echo "📊 RELATÓRIO FINAL"
echo "=================="

echo "✅ Pipeline Fase 1 concluído"
echo "✅ Otimização genética concluída"
echo "✅ Explicações LLM geradas"
echo "✅ Testes executados"

echo ""
echo "📁 Arquivos gerados:"
echo "  📊 reports/ga_*.json - Resultados da otimização genética"
echo "  📝 reports/*_explanation.txt - Explicações do LLM"
echo "  👥 reports/patient_reports.json - Relatórios para pacientes"
echo "  📈 reports/figures/ - Gráficos e visualizações"

echo ""
echo "📚 Próximos passos:"
echo "  1. Abrir notebooks/02_genetic_optimization.ipynb"
echo "  2. Abrir notebooks/03_llm_interpretation.ipynb"
echo "  3. Revisar relatórios em reports/"
echo "  4. Criar vídeo de demonstração (15 min)"

echo ""
echo "🎉 Tech Challenge Fase 2 concluído com sucesso!"