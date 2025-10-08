# ML Template FULL (Tabular, Un/Supervised, Time Series, Deep Learning)

Dieses Template bietet:
- **CLI-Workflow** (Supervised/Unsupervised/Time Series/Deep Learning)
- **Sklearn-Pipelines**, **Hyperparameter-Tuning** (Grid/RandomizedSearchCV), **CV** (inkl. TimeSeriesSplit)
- **Time Series Feature Engineering** (Lags, Rolling, Datetime, Fourier, optionale saisonale Dekomposition)
- **Forecasting-Modelle**: SARIMAX, XGBoost-Lag, *optional* Prophet
- **Deep Learning**: Tabular-MLP, LSTM-Forecaster (PyTorch)
- **MLflow**: Experimente, Metriken & Artefakte
- **DVC**: Pipeline-Skelett
- **Docker**: CPU-Image
- **Cookiecutter**: `cookiecutter.json` zur Wiederverwendung

## Quickstart
```bash
pip install -r requirements.txt
python main.py
```

Ergebnisse findest du in `reports/` und (falls aktiv) in `mlruns/`.

## MLflow
Optional: `export MLFLOW_TRACKING_URI=file:./mlruns` (Default). Runs werden automatisch geloggt.

## DVC
```bash
dvc init
dvc repro
```

## Docker
```bash
docker build -t ml_template_full .
docker run --rm -it -v $(pwd):/app ml_template_full
```

## Cookiecutter
Dieses Repo als Vorlage verwenden: `cookiecutter ./` (lokal) und Fragen beantworten.
