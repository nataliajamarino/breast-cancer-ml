"""
Módulo de integração com LLMs para explicações automáticas de diagnóstico médico.
Gera explicações em linguagem natural sobre resultados de modelos de ML.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Tentativa de importar OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI não disponível. Usando simulador local.")

# Tentativa de importar transformers para modelos locais
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers não disponível. Usando apenas OpenAI ou simulador.")


class LLMExplainer:
    """
    Gerador de explicações médicas usando LLMs.
    """
    
    def __init__(self, 
                 provider: str = "openai",
                 model_name: str = "gpt-3.5-turbo",
                 api_key: Optional[str] = None,
                 temperature: float = 0.3,
                 max_tokens: int = 500):
        """
        Inicializa o explicador LLM.
        
        Args:
            provider: Provedor do LLM ('openai', 'huggingface', 'simulator')
            model_name: Nome do modelo
            api_key: Chave da API (se necessário)
            temperature: Criatividade das respostas
            max_tokens: Máximo de tokens na resposta
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configurar provider
        if provider == "openai":
            self._setup_openai(api_key)
        elif provider == "huggingface":
            self._setup_huggingface()
        else:
            print("Usando simulador de LLM (respostas pré-definidas)")
            self.provider = "simulator"
    
    def _setup_openai(self, api_key: Optional[str]):
        """Configura cliente OpenAI."""
        if not OPENAI_AVAILABLE:
            print("OpenAI não disponível. Usando simulador.")
            self.provider = "simulator"
            return
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("API key não encontrada. Usando simulador.")
            self.provider = "simulator"
            return
        
        try:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
            print(f"OpenAI configurado: {self.model_name}")
        except Exception as e:
            print(f"Erro ao configurar OpenAI: {e}. Usando simulador.")
            self.provider = "simulator"
    
    def _setup_huggingface(self):
        """Configura modelo HuggingFace local."""
        if not TRANSFORMERS_AVAILABLE:
            print("Transformers não disponível. Usando simulador.")
            self.provider = "simulator"
            return
        
        try:
            # Modelo pequeno para demonstração
            model_name = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Adicionar pad token se não existir
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"HuggingFace configurado: {model_name}")
        except Exception as e:
            print(f"Erro ao configurar HuggingFace: {e}. Usando simulador.")
            self.provider = "simulator"
    
    def _call_openai(self, prompt: str) -> str:
        """Chama a API do OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Você é um assistente médico especializado em explicar resultados de diagnóstico de câncer de mama de forma clara e compreensível."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Erro na chamada OpenAI: {e}")
            return self._get_fallback_response(prompt)
    
    def _call_huggingface(self, prompt: str) -> str:
        """Chama modelo HuggingFace local."""
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remover o prompt da resposta
            response = response[len(prompt):].strip()
            return response
        except Exception as e:
            print(f"Erro na chamada HuggingFace: {e}")
            return self._get_fallback_response(prompt)
    
    def _get_fallback_response(self, prompt: str) -> str:
        """Resposta de fallback baseada em templates."""
        # Análise simples do prompt para gerar resposta apropriada
        prompt_lower = prompt.lower()
        
        if "diagnóstico" in prompt_lower or "prediction" in prompt_lower:
            if "maligno" in prompt_lower or "malignant" in prompt_lower:
                return """DIAGNÓSTICO: MALIGNO (Câncer)

**Explicação do Resultado:**
O modelo de machine learning analisou as características das células e identificou padrões consistentes com câncer de mama maligno. Isso significa que as células apresentam características anormais que indicam crescimento canceroso.

**Características Relevantes:**
- Tamanho e formato irregular das células
- Textura celular alterada
- Padrões de crescimento anômalos

**Próximos Passos Recomendados:**
1. Confirmação através de biópsia
2. Consulta com oncologista
3. Exames complementares para estadiamento
4. Discussão de opções de tratamento

**Importante:** Este é um resultado preliminar que deve ser confirmado por profissionais médicos qualificados."""

            else:
                return """DIAGNÓSTICO: BENIGNO (Não-canceroso)

**Explicação do Resultado:**
O modelo de machine learning analisou as características das células e identificou padrões consistentes com tecido benigno. Isso significa que não foram detectadas características indicativas de câncer.

**Características Observadas:**
- Células com formato e tamanho normais
- Textura celular dentro dos padrões esperados
- Ausência de marcadores de malignidade

**Recomendações:**
1. Acompanhamento médico regular
2. Monitoramento conforme protocolo médico
3. Atenção a mudanças futuras

**Importante:** Mesmo com resultado benigno, é essencial manter acompanhamento médico regular."""

        elif "modelo" in prompt_lower or "performance" in prompt_lower:
            return """**Análise de Performance do Modelo:**

O modelo de machine learning foi otimizado usando algoritmos genéticos para maximizar a precisão no diagnóstico de câncer de mama.

**Métricas de Desempenho:**
- **Acurácia:** Percentual de diagnósticos corretos
- **Sensibilidade (Recall):** Capacidade de detectar casos positivos
- **Especificidade:** Capacidade de identificar casos negativos
- **F1-Score:** Média harmônica entre precisão e recall

**Otimização Genética:**
O algoritmo genético testou milhares de combinações de parâmetros para encontrar a configuração que melhor equilibra todas as métricas, priorizando a detecção de casos malignos (alta sensibilidade) para evitar falsos negativos em diagnósticos médicos."""

        else:
            return """**Explicação Geral sobre Diagnóstico de Câncer de Mama com IA:**

Os modelos de machine learning analisam múltiplas características das células para fazer predições sobre a natureza do tecido (benigno ou maligno).

**Como Funciona:**
1. Análise de características celulares
2. Comparação com padrões conhecidos
3. Cálculo de probabilidades
4. Geração de diagnóstico

**Limitações:**
- Resultados devem ser validados por médicos
- IA é ferramenta de apoio, não substituto
- Sempre necessário acompanhamento profissional"""
    
    def generate_explanation(self, 
                           prediction: int,
                           probability: float,
                           feature_importance: Optional[Dict[str, float]] = None,
                           model_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Gera explicação sobre um diagnóstico específico.
        
        Args:
            prediction: Predição do modelo (0=benigno, 1=maligno)
            probability: Probabilidade da predição
            feature_importance: Importância das features (opcional)
            model_metrics: Métricas do modelo (opcional)
            
        Returns:
            Explicação em linguagem natural
        """
        # Construir prompt
        diagnosis = "maligno" if prediction == 1 else "benigno"
        confidence = "alta" if probability > 0.8 else "média" if probability > 0.6 else "baixa"
        
        prompt = f"""
Explique o seguinte diagnóstico médico de câncer de mama de forma clara e compreensível:

RESULTADO DO DIAGNÓSTICO:
- Classificação: {diagnosis}
- Probabilidade: {probability:.2%}
- Confiança: {confidence}
"""
        
        if feature_importance:
            prompt += "\nCARACTERÍSTICAS MAIS IMPORTANTES:\n"
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                prompt += f"- {feature}: {importance:.3f}\n"
        
        if model_metrics:
            prompt += "\nPERFORMANCE DO MODELO:\n"
            for metric, value in model_metrics.items():
                prompt += f"- {metric}: {value:.3f}\n"
        
        prompt += """
Forneça uma explicação que inclua:
1. O que significa o diagnóstico
2. O nível de confiança do resultado
3. Quais características foram mais importantes
4. Próximos passos recomendados
5. Limitações e importância da confirmação médica

Use linguagem acessível para pacientes e profissionais de saúde.
"""
        
        # Gerar resposta baseada no provider
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt)
        else:
            return self._get_fallback_response(prompt)
    
    def explain_model_performance(self, 
                                model_results: Dict[str, Any],
                                optimization_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Explica a performance do modelo e otimização.
        
        Args:
            model_results: Resultados da avaliação do modelo
            optimization_results: Resultados da otimização genética (opcional)
            
        Returns:
            Explicação da performance
        """
        prompt = f"""
Explique a performance do seguinte modelo de machine learning para diagnóstico de câncer de mama:

MÉTRICAS DO MODELO:
"""
        
        if 'test_metrics' in model_results:
            for metric, value in model_results['test_metrics'].items():
                prompt += f"- {metric}: {value:.3f}\n"
        
        if optimization_results:
            prompt += f"""
OTIMIZAÇÃO GENÉTICA:
- Fitness Final: {optimization_results.get('best_fitness', 'N/A'):.3f}
- Parâmetros Otimizados: {len(optimization_results.get('best_params', {}))} parâmetros
"""
        
        prompt += """
Explique:
1. O que cada métrica significa no contexto médico
2. Como a otimização genética melhorou o modelo
3. Confiabilidade do modelo para uso clínico
4. Limitações e considerações importantes

Use linguagem técnica mas acessível para profissionais de saúde.
"""
        
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt)
        else:
            return self._get_fallback_response(prompt)
    
    def compare_models(self, model_comparison: Dict[str, Dict[str, float]]) -> str:
        """
        Compara diferentes modelos e explica as diferenças.
        
        Args:
            model_comparison: Dicionário com métricas de diferentes modelos
            
        Returns:
            Comparação explicada
        """
        prompt = "Compare os seguintes modelos de diagnóstico de câncer de mama:\n\n"
        
        for model_name, metrics in model_comparison.items():
            prompt += f"MODELO: {model_name.upper()}\n"
            for metric, value in metrics.items():
                prompt += f"- {metric}: {value:.3f}\n"
            prompt += "\n"
        
        prompt += """
Forneça uma análise comparativa que inclua:
1. Qual modelo teve melhor performance geral
2. Qual modelo é mais confiável para detectar câncer (sensibilidade)
3. Qual modelo tem menor taxa de falsos positivos (especificidade)
4. Recomendação de qual modelo usar clinicamente
5. Considerações sobre trade-offs entre métricas

Foque na aplicação clínica e segurança do paciente.
"""
        
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt)
        else:
            return self._get_fallback_response(prompt)
    
    def generate_patient_report(self, 
                              patient_data: Dict[str, Any],
                              prediction_results: Dict[str, Any]) -> str:
        """
        Gera relatório completo para paciente.
        
        Args:
            patient_data: Dados do paciente (anonimizados)
            prediction_results: Resultados da predição
            
        Returns:
            Relatório para paciente
        """
        diagnosis = "maligno" if prediction_results['prediction'] == 1 else "benigno"
        probability = prediction_results['probability']
        
        prompt = f"""
Gere um relatório médico para paciente sobre diagnóstico de câncer de mama:

RESULTADO:
- Diagnóstico: {diagnosis}
- Confiança: {probability:.1%}

Crie um relatório que:
1. Explique o resultado de forma clara e tranquilizadora
2. Descreva o que o diagnóstico significa
3. Liste próximos passos recomendados
4. Forneça informações sobre confiabilidade do teste
5. Inclua mensagem de apoio apropriada
6. Enfatize importância do acompanhamento médico

Use linguagem empática e acessível para leigos.
"""
        
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "huggingface":
            return self._call_huggingface(prompt)
        else:
            return self._get_fallback_response(prompt)


def demo_llm_explanations():
    """
    Demonstração das funcionalidades do LLM Explainer.
    """
    print("=== DEMO: LLM Explainer ===\n")
    
    # Inicializar explainer (tentará OpenAI, depois HuggingFace, por fim simulador)
    explainer = LLMExplainer(provider="simulator")  # Usar simulador para demo
    
    # Exemplo 1: Diagnóstico maligno
    print("1. EXPLICAÇÃO DE DIAGNÓSTICO MALIGNO:")
    print("-" * 50)
    explanation1 = explainer.generate_explanation(
        prediction=1,
        probability=0.89,
        feature_importance={
            'texture_mean': 0.23,
            'area_mean': 0.19,
            'smoothness_mean': 0.15,
            'compactness_mean': 0.12,
            'concavity_mean': 0.11
        },
        model_metrics={
            'accuracy': 0.95,
            'recall': 0.97,
            'precision': 0.93,
            'f1_score': 0.95
        }
    )
    print(explanation1)
    
    print("\n" + "="*80 + "\n")
    
    # Exemplo 2: Diagnóstico benigno
    print("2. EXPLICAÇÃO DE DIAGNÓSTICO BENIGNO:")
    print("-" * 50)
    explanation2 = explainer.generate_explanation(
        prediction=0,
        probability=0.92,
        feature_importance={
            'texture_mean': 0.18,
            'perimeter_mean': 0.16,
            'area_mean': 0.14,
            'smoothness_mean': 0.12,
            'symmetry_mean': 0.10
        }
    )
    print(explanation2)
    
    print("\n" + "="*80 + "\n")
    
    # Exemplo 3: Comparação de modelos
    print("3. COMPARAÇÃO DE MODELOS:")
    print("-" * 50)
    comparison = explainer.compare_models({
        'random_forest': {
            'accuracy': 0.94,
            'recall': 0.96,
            'precision': 0.92,
            'f1_score': 0.94
        },
        'xgboost': {
            'accuracy': 0.96,
            'recall': 0.95,
            'precision': 0.97,
            'f1_score': 0.96
        },
        'svm': {
            'accuracy': 0.93,
            'recall': 0.98,
            'precision': 0.89,
            'f1_score': 0.93
        }
    })
    print(comparison)
    
    return {
        'malignant_explanation': explanation1,
        'benign_explanation': explanation2,
        'model_comparison': comparison
    }


if __name__ == "__main__":
    demo_llm_explanations()