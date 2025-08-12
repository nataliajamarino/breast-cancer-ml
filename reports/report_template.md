
# Breast Cancer Prediction — Project Report (Template)

## 1. Problem & Data
- Task: Binary classification (malignant vs. benign). Dataset: Breast Cancer Wisconsin (Diagnostic).
- Rationale: optimize **recall** (sensitivity) to reduce false negatives.

## 2. Methodology
- Data splits: stratified Train/Val/Test.
- Models: Logistic Regression, SVM (RBF), Random Forest, XGBoost, LightGBM.
- Calibration: isotonic for reliable probabilities.
- Thresholding: choose the threshold that achieves recall ≥ 0.95 with best precision.

## 3. Results
- See `reports/leaderboard.json` for per-model metrics.
- Curves: `reports/figures/roc_*.png`, `reports/figures/pr_*.png`.
- Per-model details in `reports/eval_<model>.json`.

## 4. Explainability
- Start with feature importances (tree-based) or coefficients (logistic). Add SHAP if needed.

## 5. Conclusion & Next Steps
- Discuss trade-offs between recall and precision and the clinical workflow (physician in-the-loop).
- Potential bonus: mammography CNN, active learning, or cost-sensitive learning.

