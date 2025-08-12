
# Data folder

This project is now configured to **use `data.csv` by default** if present.
Expected columns:
- `diagnosis` (target): 'M' (malignant) or 'B' (benign) — will be mapped to 1/0
- 30 numeric features (e.g., `radius_mean`, `texture_mean`, ..., `fractal_dimension_worst`)
- Optional columns `id`, `Unnamed: 32` will be ignored.

If `data.csv` is absent, the pipeline falls back to scikit-learn's built-in WDBC dataset.
