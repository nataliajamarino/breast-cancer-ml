
from __future__ import annotations
from typing import Dict, Any, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def build_pipelines(n_features: int) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    # For this dataset, all features are numeric; scale is helpful for SVM/LogReg.
    scaler = StandardScaler()
    pre = ColumnTransformer([('scaler', scaler, list(range(n_features)))], remainder='drop')

    pipelines: Dict[str, Tuple[Pipeline, Dict[str, Any]]] = {}

    # Logistic Regression (with L2)
    logreg = Pipeline([
        ('pre', pre),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    logreg_params = {
        'clf__C': [0.1, 1.0, 10.0]
    }
    pipelines['logreg'] = (logreg, logreg_params)

    # SVM RBF
    svm = Pipeline([
        ('pre', pre),
        ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced'))
    ])
    svm_params = {
        'clf__C': [0.5, 1.0, 5.0, 10.0],
        'clf__gamma': ['scale', 'auto']
    }
    pipelines['svm_rbf'] = (svm, svm_params)

    # Random Forest
    rf = Pipeline([
        ('pre', pre),  # not strictly needed, but keeps interface consistent
        ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=600, random_state=42))
    ])
    rf_params = {
        'clf__max_depth': [None, 10, 20, 30],
        'clf__min_samples_leaf': [1, 2, 4]
    }
    pipelines['random_forest'] = (rf, rf_params)

    # XGBoost
    xgb = Pipeline([
        ('pre', pre),
        ('clf', XGBClassifier(
            n_estimators=800,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            eval_metric='logloss',
            random_state=42,
            tree_method='hist'
        ))
    ])
    xgb_params = {
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.03, 0.05, 0.1],
        'clf__n_estimators': [600, 800, 1000]
    }
    pipelines['xgboost'] = (xgb, xgb_params)

    # LightGBM
    lgbm = Pipeline([
        ('pre', pre),
        ('clf', LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42
        ))
    ])
    lgbm_params = {
        'clf__max_depth': [-1, 5, 7, 9],
        'clf__learning_rate': [0.03, 0.05, 0.1],
        'clf__n_estimators': [600, 800, 1000]
    }
    pipelines['lightgbm'] = (lgbm, lgbm_params)

    return pipelines
