# ML Template (Tabular, Un/Supervised & Time Series)

Ein generisches, modular aufgebautes ML-Template mit:
- **CLI-Workflow** (Auswahl: Supervised, Unsupervised, Time Series Forecasting)
- **Pipelines** (Sklearn)
- **Hyperparameter-Tuning** (Grid/RandomizedSearchCV)
- **Cross-Validation** (inkl. TimeSeriesSplit)
- **Evaluation & Reports** (speichert Metriken & Modelle)
- **Time Series Feature Engineering** (Lags, Rolling, Datetime, Fourier, saisonale Dekomposition)

## Schneller Start

```bash
pip install -r requirements.txt
python main.py
```

Das CLI fragt dich, was du tun willst (z. B. Forecasting), und führt die passende Pipeline aus.
Datenpfade & Konfiguration in `config/config.yaml` anpassen.

## Ordnerstruktur

```
ml_template/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── exploration.ipynb
├── reports/
├── src/
│   ├── data_loader.py
│   ├── cli.py
│   ├── preprocessing/
│   │   ├── tabular_preprocessing.py
│   │   ├── timeseries_preprocessing.py
│   │   └── feature_engineering.py
│   ├── modeling/
│   │   ├── supervised_models.py
│   │   ├── unsupervised_models.py
│   │   ├── timeseries_models.py
│   │   └── training.py
│   ├── evaluation/
│   │   ├── metrics_supervised.py
│   │   ├── metrics_unsupervised.py
│   │   └── metrics_timeseries.py
│   └── utils/
│       ├── logger.py
│       └── config_handler.py
├── main.py
└── requirements.txt
```

## Hinweise
- Prophet ist optional – standardmäßig ist **SARIMAX** & **XGBoost** integriert.
- Beispiel-Parametergrids sind enthalten; passe sie an deine Daten an.
