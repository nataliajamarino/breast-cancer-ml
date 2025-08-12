#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, json
import pandas as pd
from joblib import load

# As 30 features padrão do WDBC, na ordem esperada
WDBC_FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst",
]

MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

def load_best_model_and_threshold():
    # Pega o 1º do leaderboard (já está ordenado do melhor pro pior)
    with open(os.path.join(REPORTS_DIR, "leaderboard.json")) as f:
        leaderboard = json.load(f)
    if not leaderboard:
        raise RuntimeError("leaderboard.json está vazio. Rode a avaliação primeiro (python src/evaluate.py).")
    best = leaderboard[0]
    model_name = best["model"]
    threshold  = float(best["threshold"])

    # Descobre o caminho do modelo no registry
    with open(os.path.join(MODELS_DIR, "registry.json")) as f:
        registry = json.load(f)
    model_path = os.path.join(MODELS_DIR, os.path.basename(registry[model_name]["model_path"]))
    model = load(model_path)
    return model, threshold, model_name

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Remove colunas não-features, se existirem
    for col in ["id", "diagnosis", "Unnamed: 32"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    # Garante que temos exatamente as 30 colunas, na ordem certa
    missing = [c for c in WDBC_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias: {missing}")
    return df[WDBC_FEATURES].copy()

def main():
    ap = argparse.ArgumentParser(description="Predizer maligno/benigno em novos casos (CSV com 30 features WDBC).")
    ap.add_argument("--input",  required=True, help="Caminho do CSV com as 30 features.")
    ap.add_argument("--output", required=True, help="Arquivo de saída (CSV) com probabilidades e rótulos.")
    args = ap.parse_args()

    model, threshold, model_name = load_best_model_and_threshold()

    df = pd.read_csv(args.input)
    X  = prepare_features(df)
    proba = model.predict_proba(X)[:, 1]          # probabilidade de maligno
    pred  = (proba >= threshold).astype(int)      # aplica o threshold escolhido

    out = df.copy()
    out["proba_malignant"]   = proba
    out["pred_malignant"]    = pred
    out["decision_threshold"] = threshold
    out["model"]              = model_name

    out.to_csv(args.output, index=False)
    print(f"OK: {args.output} (modelo={model_name}, threshold={threshold:.6f})")

if __name__ == "__main__":
    main()
