
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score, roc_auc_score, average_precision_score
from features import load_dataset, train_val_test_split
from models_tabular import build_pipelines

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(OUT_MODELS, exist_ok=True)

def main():
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    pipelines = build_pipelines(X.shape[1])

    # CV focused on recall; we still collect other metrics for reference.
    scoring = {
        'recall': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score, needs_threshold=True),
        'pr_auc': make_scorer(average_precision_score, needs_threshold=True)
    }
    best_models = {}
    for name, (pipe, params) in pipelines.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            scoring=scoring,
            refit='recall',  # prioritize recall
            cv=cv,
            n_jobs=-1,
            verbose=0,
            return_train_score=False
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_

        # Probability calibration (isotonic) for better thresholding
        calibrator = CalibratedClassifierCV(best, method='isotonic', cv=5)
        calibrator.fit(X_val, y_val)  # calibrate on validation set

        # Save
        model_path = os.path.join(OUT_MODELS, f"{name}_calibrated.joblib")
        dump(calibrator, model_path)
        best_models[name] = {
            'best_params': gs.best_params_,
            'cv_best_recall': float(gs.best_score_),
            'model_path': os.path.relpath(model_path, start=os.path.join(os.path.dirname(__file__), '..'))
        }
        print(f"[saved] {name}: {model_path}")

    # Save a registry.json so evaluate.py knows which models to compare
    with open(os.path.join(OUT_MODELS, 'registry.json'), 'w') as f:
        json.dump(best_models, f, indent=2)

    print("Training complete. Models saved in models/ with calibration.")

if __name__ == '__main__':
    main()
