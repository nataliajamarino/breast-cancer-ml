"""
Testes automatizados para o otimizador genético.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Adicionar src ao path para testes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ga_optimizer import GeneticOptimizer
from llm_explainer import LLMExplainer


class TestGeneticOptimizer:
    """Testes para a classe GeneticOptimizer."""
    
    @pytest.fixture
    def sample_data(self):
        """Cria dataset sintético para testes."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def optimizer(self):
        """Cria instância do otimizador para testes."""
        return GeneticOptimizer(
            model_type='random_forest',
            population_size=10,
            generations=5,
            random_state=42
        )
    
    def test_optimizer_initialization(self):
        """Testa inicialização do otimizador."""
        optimizer = GeneticOptimizer()
        assert optimizer.model_type == 'random_forest'
        assert optimizer.population_size == 50
        assert optimizer.generations == 30
        assert optimizer.random_state == 42
    
    def test_param_spaces_definition(self, optimizer):
        """Testa definição dos espaços de parâmetros."""
        assert 'random_forest' in optimizer.param_spaces
        assert 'xgboost' in optimizer.param_spaces
        assert 'lightgbm' in optimizer.param_spaces
        assert 'svm' in optimizer.param_spaces
        assert 'logistic' in optimizer.param_spaces
        
        # Verifica parâmetros do Random Forest
        rf_params = optimizer.param_spaces['random_forest']
        assert 'n_estimators' in rf_params
        assert 'max_depth' in rf_params
        assert 'min_samples_split' in rf_params
    
    def test_individual_creation(self, optimizer):
        """Testa criação de indivíduos."""
        individual = optimizer._create_individual()
        assert len(individual) == len(optimizer.param_spaces['random_forest'])
        assert hasattr(individual, 'fitness')
    
    def test_individual_to_params_conversion(self, optimizer):
        """Testa conversão de indivíduo para parâmetros."""
        individual = optimizer._create_individual()
        params = optimizer._individual_to_params(individual)
        
        expected_params = list(optimizer.param_spaces['random_forest'].keys())
        assert all(param in params for param in expected_params)
        
        # Verifica tipos dos parâmetros
        assert isinstance(params['n_estimators'], int)
        assert isinstance(params['max_depth'], int)
        assert params['max_features'] in ['sqrt', 'log2', None]
    
    def test_model_creation(self, optimizer):
        """Testa criação de modelos."""
        individual = optimizer._create_individual()
        params = optimizer._individual_to_params(individual)
        model = optimizer._create_model(params)
        
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == optimizer.random_state
    
    def test_optimization_execution(self, optimizer, sample_data):
        """Testa execução da otimização."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Executar otimização rápida
        result = optimizer.optimize(X_train, y_train, verbose=False)
        
        # Verificar estrutura dos resultados
        assert 'best_params' in result
        assert 'best_model' in result
        assert 'best_fitness' in result
        assert 'history' in result
        assert 'convergence_data' in result
        
        # Verificar qualidade dos resultados
        assert result['best_fitness'] > 0
        assert result['best_fitness'] <= 1.0
        assert len(result['convergence_data']['generation']) == optimizer.generations
    
    def test_different_models(self, sample_data):
        """Testa otimização com diferentes modelos."""
        X_train, X_test, y_train, y_test = sample_data
        models = ['random_forest', 'xgboost', 'lightgbm', 'svm', 'logistic']
        
        for model_type in models:
            optimizer = GeneticOptimizer(
                model_type=model_type,
                population_size=8,
                generations=3,
                random_state=42
            )
            
            try:
                result = optimizer.optimize(X_train, y_train, verbose=False)
                assert result['best_fitness'] > 0
                print(f"✅ {model_type}: fitness = {result['best_fitness']:.3f}")
            except Exception as e:
                pytest.fail(f"Erro na otimização do modelo {model_type}: {e}")
    
    def test_convergence_improvement(self, optimizer, sample_data):
        """Testa se o algoritmo genético melhora ao longo das gerações."""
        X_train, X_test, y_train, y_test = sample_data
        
        result = optimizer.optimize(X_train, y_train, verbose=False)
        convergence = result['convergence_data']
        
        # Fitness deve melhorar ou se manter
        initial_fitness = convergence['max_fitness'][0]
        final_fitness = convergence['max_fitness'][-1]
        
        assert final_fitness >= initial_fitness, "Fitness não melhorou"
        print(f"Melhoria: {initial_fitness:.3f} → {final_fitness:.3f}")
    
    def test_save_and_load_results(self, optimizer, sample_data, tmp_path):
        """Testa salvamento e carregamento dos resultados."""
        X_train, X_test, y_train, y_test = sample_data
        
        result = optimizer.optimize(X_train, y_train, verbose=False)
        
        # Salvar resultados
        save_path = tmp_path / "test_results.json"
        optimizer.save_results(result, str(save_path))
        
        # Verificar se arquivos foram criados
        assert save_path.exists()
        model_path = tmp_path / "test_results_model.joblib"
        assert model_path.exists()
        
        # Carregar e verificar
        import json
        import joblib
        
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['model_type'] == optimizer.model_type
        assert saved_data['best_fitness'] == result['best_fitness']
        
        loaded_model = joblib.load(model_path)
        assert loaded_model is not None


class TestLLMExplainer:
    """Testes para a classe LLMExplainer."""
    
    @pytest.fixture
    def explainer(self):
        """Cria instância do explainer para testes."""
        return LLMExplainer(provider="simulator")
    
    def test_explainer_initialization(self):
        """Testa inicialização do explainer."""
        explainer = LLMExplainer(provider="simulator")
        assert explainer.provider == "simulator"
        assert explainer.temperature == 0.3
        assert explainer.max_tokens == 500
    
    def test_generate_explanation_malignant(self, explainer):
        """Testa geração de explicação para caso maligno."""
        explanation = explainer.generate_explanation(
            prediction=1,
            probability=0.89,
            feature_importance={'texture_mean': 0.2, 'area_mean': 0.15},
            model_metrics={'accuracy': 0.95, 'recall': 0.97}
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 100
        assert 'maligno' in explanation.lower() or 'malignant' in explanation.lower()
        print(f"Explicação maligna gerada: {len(explanation)} caracteres")
    
    def test_generate_explanation_benign(self, explainer):
        """Testa geração de explicação para caso benigno."""
        explanation = explainer.generate_explanation(
            prediction=0,
            probability=0.92,
            feature_importance={'texture_mean': 0.18, 'area_mean': 0.14}
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 100
        assert 'benigno' in explanation.lower() or 'benign' in explanation.lower()
        print(f"Explicação benigna gerada: {len(explanation)} caracteres")
    
    def test_model_performance_explanation(self, explainer):
        """Testa explicação de performance do modelo."""
        model_results = {
            'test_metrics': {
                'accuracy': 0.95,
                'precision': 0.93,
                'recall': 0.97,
                'f1_score': 0.95
            }
        }
        
        explanation = explainer.explain_model_performance(model_results)
        assert isinstance(explanation, str)
        assert len(explanation) > 100
        print(f"Explicação de performance: {len(explanation)} caracteres")
    
    def test_compare_models(self, explainer):
        """Testa comparação de modelos."""
        model_comparison = {
            'random_forest': {
                'accuracy': 0.94,
                'recall': 0.96,
                'precision': 0.92
            },
            'xgboost': {
                'accuracy': 0.96,
                'recall': 0.95,
                'precision': 0.97
            }
        }
        
        comparison = explainer.compare_models(model_comparison)
        assert isinstance(comparison, str)
        assert len(comparison) > 100
        print(f"Comparação de modelos: {len(comparison)} caracteres")
    
    def test_patient_report_generation(self, explainer):
        """Testa geração de relatório para paciente."""
        patient_data = {'case_id': 'TEST001', 'age_group': '40-49 anos'}
        prediction_results = {'prediction': 0, 'probability': 0.85}
        
        report = explainer.generate_patient_report(patient_data, prediction_results)
        assert isinstance(report, str)
        assert len(report) > 100
        print(f"Relatório do paciente: {len(report)} caracteres")
    
    def test_fallback_responses(self, explainer):
        """Testa respostas de fallback."""
        # Teste com diferentes tipos de prompts
        prompts = [
            "diagnóstico maligno",
            "diagnóstico benigno", 
            "performance do modelo",
            "prompt genérico"
        ]
        
        for prompt in prompts:
            response = explainer._get_fallback_response(prompt)
            assert isinstance(response, str)
            assert len(response) > 50
            print(f"Prompt '{prompt}': {len(response)} caracteres")


class TestIntegration:
    """Testes de integração entre componentes."""
    
    def test_ga_optimization_with_llm_explanation(self):
        """Testa integração completa: GA + LLM."""
        # Criar dados sintéticos
        X, y = make_classification(
            n_samples=100,
            n_features=8,
            n_informative=6,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Otimizar modelo
        optimizer = GeneticOptimizer(
            model_type='random_forest',
            population_size=8,
            generations=3,
            random_state=42
        )
        
        ga_result = optimizer.optimize(X_train, y_train, verbose=False)
        
        # Avaliar no teste
        model = ga_result['best_model']
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Gerar explicação
        explainer = LLMExplainer(provider="simulator")
        
        # Explicar performance
        performance_explanation = explainer.explain_model_performance(
            model_results={'test_metrics': metrics},
            optimization_results=ga_result
        )
        
        # Explicar predição individual
        idx = 0
        individual_explanation = explainer.generate_explanation(
            prediction=y_pred[idx],
            probability=y_prob[idx],
            model_metrics=metrics
        )
        
        # Verificações
        assert ga_result['best_fitness'] > 0
        assert len(performance_explanation) > 100
        assert len(individual_explanation) > 100
        
        print(f"✅ Integração completa testada:")
        print(f"  - GA Fitness: {ga_result['best_fitness']:.3f}")
        print(f"  - Acurácia: {metrics['accuracy']:.3f}")
        print(f"  - Explicações geradas com sucesso")


def test_demo_functions():
    """Testa funções de demonstração."""
    from llm_explainer import demo_llm_explanations
    
    # Executar demo
    demo_results = demo_llm_explanations()
    
    assert 'malignant_explanation' in demo_results
    assert 'benign_explanation' in demo_results
    assert 'model_comparison' in demo_results
    
    for key, explanation in demo_results.items():
        assert isinstance(explanation, str)
        assert len(explanation) > 100
    
    print("✅ Funções de demonstração testadas")


if __name__ == "__main__":
    # Executar testes quando rodado diretamente
    pytest.main([__file__, "-v", "--tb=short"])