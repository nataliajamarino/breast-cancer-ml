
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
)
from features import load_dataset, train_val_test_split
from utils import find_threshold_for_recall

RANDOM_STATE = 42

OUT_MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_REPORTS = os.path.join(os.path.dirname(__file__), "..", "reports")
FIG_DIR = os.path.join(OUT_REPORTS, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_roc_pr(y_true, y_proba, tag: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {tag}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'roc_{tag}.png'))
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {tag}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'pr_{tag}.png'))
    plt.close()

def main():
    # Load data
    X, y = load_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    # Load registry
    with open(os.path.join(OUT_MODELS, 'registry.json')) as f:
        registry = json.load(f)

    summary = []
    for name, meta in registry.items():
        model = load(os.path.join(OUT_MODELS, os.path.basename(meta['model_path'])))
        # Evaluate on test
        y_proba = model.predict_proba(X_test)[:, 1]
        tag = name

        # Curves
        plot_roc_pr(y_test.values, y_proba, tag)

        # Threshold selection for recall >= 0.95
        thr = find_threshold_for_recall(y_test.values, y_proba, min_recall=0.95)
        y_pred = (y_proba >= thr.threshold).astype(int)

        cm = confusion_matrix(y_test.values, y_pred)
        report = classification_report(y_test.values, y_pred, digits=4, output_dict=True)

        # Save per-model json
        model_report = {
            'model': name,
            'threshold': thr.threshold,
            'recall': thr.recall,
            'precision': thr.precision,
            'f1': thr.f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        with open(os.path.join(OUT_REPORTS, f'eval_{name}.json'), 'w') as f:
            json.dump(model_report, f, indent=2)

        summary.append({
            'model': name,
            **{k: v for k, v in model_report.items() if k in ['threshold','recall','precision','f1']}
        })

    # Save leaderboard
    summary = sorted(summary, key=lambda d: (-d['recall'], -d['precision'], -d['f1']))
    with open(os.path.join(OUT_REPORTS, 'leaderboard.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print('Evaluation complete. See reports/ for JSON and figures/.')

if __name__ == '__main__':
    main()
