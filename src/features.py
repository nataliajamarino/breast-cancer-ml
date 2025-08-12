
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pathlib import Path

RANDOM_STATE = 42
DATA_CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "data.csv"

def load_from_csv(csv_path: Path | str = DATA_CSV_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load a CSV with columns like the WDBC dataset.
    Expected:
      - target column: 'diagnosis' with values 'M'/'B' (or 1/0)
      - potential extra columns: 'id', 'Unnamed: 32' (will be dropped if present)
    Returns X (DataFrame) and y (Series) with y mapped to 1=malignant, 0=benign.
    """
    df = pd.read_csv(csv_path)
    # Remove columns that aren't features
    drop_cols = [c for c in ['id', 'Unnamed: 32'] if c in df.columns]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Standardize target
    if 'diagnosis' not in df.columns:
        raise ValueError("CSV must contain a 'diagnosis' column (values 'M'/'B' or 1/0).")
    y_raw = df['diagnosis']
    if y_raw.dtype == object:
        # Map 'M'/'B' -> 1/0
        y = y_raw.map({'M': 1, 'B': 0}).astype(int)
    else:
        # Assume already 0/1 where 1 means malignant, 0 benign
        y = y_raw.astype(int)

    X = df.drop(columns=['diagnosis'])
    return X, y

def load_wdbc() -> tuple[pd.DataFrame, pd.Series]:
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')  # 0 = malignant, 1 = benign (per sklearn)
    # To align with clinical convention, map to 1=malignant, 0=benign
    y = (1 - y).astype(int)
    return X, y

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Prefer CSV if present; otherwise fall back to sklearn's built-in WDBC."""
    if DATA_CSV_PATH.exists():
        return load_from_csv(DATA_CSV_PATH)
    return load_wdbc()

def train_val_test_split(X, y, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    # First split train/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    # Split train into train/val
    val_rel = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_rel, stratify=y_trainval, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
