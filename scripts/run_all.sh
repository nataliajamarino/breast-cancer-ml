
#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Training models (tabular)..."
python -u src/train_tabular.py

echo "[2/3] Evaluating best model on test set..."
python -u src/evaluate.py

echo "[3/3] Done. See reports/figures and models/ for outputs."
