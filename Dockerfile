
# NOTE: building LightGBM from source may require extra build tools.
# If you hit build issues, comment out lightgbm in requirements or add build deps.
FROM python:3.11-slim

WORKDIR /app

# System deps for xgboost/lightgbm (minimal; may need more on some hosts)
RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gcc g++ git     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command: run end-to-end training & evaluation
CMD ["bash", "scripts/run_all.sh"]
