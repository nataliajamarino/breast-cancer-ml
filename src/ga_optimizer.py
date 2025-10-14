"""
Otimizador de hiperparâmetros usando Algoritmos Genéticos.
Este módulo implementa um GA para otimizar modelos de ML para diagnóstico médico.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import xgboost as xgb
import lightgbm as lgb
from deap import base, creator, tools, algorithms
import random
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Criar classes de fitness e indivíduo globalmente
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class GeneticOptimizer:
    """
    Otimizador de hiperparâmetros usando Algoritmos Genéticos.
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 population_size: int = 50,
                 generations: int = 30,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.3,
                 tournament_size: int = 3,
                 random_state: int = 42):
        """
        Inicializa o otimizador genético.
        
        Args:
            model_type: Tipo do modelo ('random_forest', 'xgboost', 'lightgbm', 'svm', 'logistic')
            population_size: Tamanho da população
            generations: Número de gerações
            crossover_prob: Probabilidade de cruzamento
            mutation_prob: Probabilidade de mutação
            tournament_size: Tamanho do torneio para seleção
            random_state: Seed para reprodutibilidade
        """
        self.model_type = model_type
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.random_state = random_state
        
        # Configurar seeds
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Definir espaços de hiperparâmetros
        self.param_spaces = self._define_param_spaces()
        
        # Configurar DEAP
        self._setup_deap()
        
        # Histórico da otimização
        self.history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
    def _define_param_spaces(self) -> Dict[str, Dict]:
        """Define os espaços de hiperparâmetros para cada modelo."""
        spaces = {
            'random_forest': {
                'n_estimators': (50, 500),
                'max_depth': (3, 30),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'xgboost': {
                'n_estimators': (50, 500),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0.0, 1.0),
                'reg_lambda': (0.0, 1.0)
            },
            'lightgbm': {
                'n_estimators': (50, 500),
                'max_depth': (3, 15),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0.0, 1.0),
                'reg_lambda': (0.0, 1.0),
                'num_leaves': (10, 300)
            },
            'svm': {
                'C': (0.1, 100.0),
                'gamma': (0.001, 1.0),
                'kernel': ['rbf', 'linear', 'poly'],
                'degree': (2, 5)  # apenas para kernel poly
            },
            'logistic': {
                'C': (0.01, 100.0),
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'max_iter': (100, 1000)
            }
        }
        return spaces
    
    def _setup_deap(self):
        """Configura o framework DEAP para algoritmos genéticos."""
        # Toolbox
        self.toolbox = base.Toolbox()
        
        # Registrar funções genéticas
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

    def _create_individual(self) -> creator.Individual:
        """Cria um indivíduo (conjunto de hiperparâmetros) aleatório."""
        individual = []
        param_space = self.param_spaces[self.model_type]
        
        for param_name, param_range in param_space.items():
            if isinstance(param_range, tuple):
                if isinstance(param_range[0], int):
                    # Parâmetro inteiro
                    value = random.randint(param_range[0], param_range[1])
                else:
                    # Parâmetro float
                    value = random.uniform(param_range[0], param_range[1])
            elif isinstance(param_range, list):
                # Parâmetro categórico
                value = random.choice(param_range)
            individual.append(value)
        
        return creator.Individual(individual)
    
    def _individual_to_params(self, individual: List) -> Dict[str, Any]:
        """Converte um indivíduo em dicionário de hiperparâmetros."""
        param_space = self.param_spaces[self.model_type]
        param_names = list(param_space.keys())
        
        params = {}
        for i, (param_name, value) in enumerate(zip(param_names, individual)):
            params[param_name] = value
        
        # Ajustes específicos por modelo
        if self.model_type == 'svm' and params.get('kernel') != 'poly':
            params.pop('degree', None)
        elif self.model_type == 'logistic':
            if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                params['solver'] = 'liblinear'
            elif params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
                params['solver'] = 'saga'
        
        return params
    
    def _create_model(self, params: Dict[str, Any]):
        """Cria um modelo com os hiperparâmetros especificados."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, **params)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(random_state=self.random_state, **params)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(random_state=self.random_state, verbose=-1, **params)
        elif self.model_type == 'svm':
            return SVC(random_state=self.random_state, **params)
        elif self.model_type == 'logistic':
            return LogisticRegression(random_state=self.random_state, **params)
        else:
            raise ValueError(f"Modelo não suportado: {self.model_type}")
    
    def _evaluate_individual(self, individual: List) -> Tuple[float]:
        """Avalia um indivíduo usando cross-validation otimizada."""
        try:
            params = self._individual_to_params(individual)
            model = self._create_model(params)

            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

            # Executa todas as métricas de uma vez
            scoring = {'accuracy': 'accuracy', 'recall': 'recall', 'f1': 'f1'}
            scores = cross_validate(model, self.X_train, self.y_train, cv=cv, scoring=scoring, n_jobs=-1)

            fitness = (
                0.4 * np.mean(scores['test_accuracy']) +
                0.4 * np.mean(scores['test_recall']) +
                0.2 * np.mean(scores['test_f1'])
            )

            return (fitness,)

        except Exception as e:
            # Penalizar indivíduos inválidos
            return (-1.0,)
    
    def _crossover(self, ind1: creator.Individual, ind2: creator.Individual) -> Tuple[creator.Individual, creator.Individual]:
        """Operador de cruzamento (crossover) uniforme."""
        param_space = self.param_spaces[self.model_type]
        
        for i, (param_name, param_range) in enumerate(param_space.items()):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        
        return ind1, ind2
    
    def _mutate(self, individual: creator.Individual) -> Tuple[creator.Individual]:
        """Operador de mutação."""
        param_space = self.param_spaces[self.model_type]
        
        for i, (param_name, param_range) in enumerate(param_space.items()):
            if random.random() < self.mutation_prob:
                if isinstance(param_range, tuple):
                    if isinstance(param_range[0], int):
                        # Mutação para inteiros
                        individual[i] = random.randint(param_range[0], param_range[1])
                    else:
                        # Mutação para floats
                        individual[i] = random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    # Mutação para categóricos
                    individual[i] = random.choice(param_range)
        
        return (individual,)
    
    def optimize(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> Dict[str, Any]:
        """
        Executa a otimização genética.
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            verbose: Se deve imprimir progresso
            
        Returns:
            Dict com resultados da otimização
        """
        self.X_train = X_train
        self.y_train = y_train
        
        # Criar população inicial
        population = self.toolbox.population(n=self.population_size)
        
        # Avaliar população inicial
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Estatísticas
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of Fame para manter os melhores
        hof = tools.HallOfFame(1)
        
        if verbose:
            print(f"Iniciando otimização genética para {self.model_type}")
            print(f"População: {self.population_size}, Gerações: {self.generations}")
        
        # Executar algoritmo genético
        population, logbook = algorithms.eaSimple(
            population, self.toolbox, 
            cxpb=self.crossover_prob, 
            mutpb=self.mutation_prob, 
            ngen=self.generations,
            stats=stats, 
            halloffame=hof, 
            verbose=verbose
        )
        
        # Salvar melhor indivíduo
        self.best_individual = hof[0]
        self.best_fitness = hof[0].fitness.values[0]
        self.history = logbook
        
        # Converter para parâmetros
        best_params = self._individual_to_params(self.best_individual)
        
        # Criar modelo final
        best_model = self._create_model(best_params)
        best_model.fit(X_train, y_train)
        
        return {
            'best_params': best_params,
            'best_model': best_model,
            'best_fitness': self.best_fitness,
            'history': logbook,
            'convergence_data': self._extract_convergence_data(logbook)
        }
    
    def _extract_convergence_data(self, logbook) -> Dict[str, List]:
        """Extrai dados de convergência para visualização."""
        return {
            'generation': list(range(len(logbook))),
            'avg_fitness': [record['avg'] for record in logbook],
            'max_fitness': [record['max'] for record in logbook],
            'min_fitness': [record['min'] for record in logbook],
            'std_fitness': [record['std'] for record in logbook]
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Salva os resultados da otimização."""
        # Preparar dados para serialização
        save_data = {
            'model_type': self.model_type,
            'best_params': results['best_params'],
            'best_fitness': results['best_fitness'],
            'convergence_data': results['convergence_data'],
            'ga_config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'crossover_prob': self.crossover_prob,
                'mutation_prob': self.mutation_prob,
                'tournament_size': self.tournament_size,
                'random_state': self.random_state
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Salvar JSON
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        # Salvar modelo
        model_filepath = filepath.replace('.json', '_model.joblib')
        joblib.dump(results['best_model'], model_filepath)
        
        print(f"Resultados salvos em: {filepath}")
        print(f"Modelo salvo em: {model_filepath}")


def run_genetic_optimization_experiments(X_train, y_train, models=['random_forest'], 
                                       experiments_config=None, verbose=True):
    """
    Executa experimentos de otimização genética para múltiplos modelos.
    
    Args:
        X_train: Features de treino
        y_train: Target de treino  
        models: Lista de modelos para otimizar
        experiments_config: Configurações dos experimentos
        verbose: Se deve imprimir progresso
        
    Returns:
        Dict com resultados de todos os experimentos
    """
    if experiments_config is None:
        experiments_config = [
            {
                'name': 'conservative',
                'population_size': 30,
                'generations': 20,
                'crossover_prob': 0.6,
                'mutation_prob': 0.2
            },
            {
                'name': 'standard',
                'population_size': 50,
                'generations': 30,
                'crossover_prob': 0.7,
                'mutation_prob': 0.3
            },
            {
                'name': 'aggressive',
                'population_size': 80,
                'generations': 50,
                'crossover_prob': 0.8,
                'mutation_prob': 0.4
            }
        ]
    
    results = {}
    
    for model_type in models:
        if verbose:
            print(f"\n🤖 Otimizando modelo: {model_type}")
            print("=" * 50)
        
        model_results = {}
        
        for config in experiments_config:
            if verbose:
                print(f"\n🧪 Experimento: {config['name']}")
            
            optimizer = GeneticOptimizer(
                model_type=model_type,
                population_size=config['population_size'],
                generations=config['generations'],
                crossover_prob=config['crossover_prob'],
                mutation_prob=config['mutation_prob']
            )
            
            result = optimizer.optimize(X_train, y_train, verbose=verbose)
            model_results[config['name']] = result
            
            if verbose:
                print(f"✅ Melhor fitness: {result['best_fitness']:.4f}")
        
        results[model_type] = model_results
    
    return results