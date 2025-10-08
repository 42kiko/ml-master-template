"""Zeitreihenanalyse und Forecasting.

Dieses Modul implementiert eine einfache Zeitreihen‑Forecasting‑Funktion auf
Basis der `statsmodels`‑ARIMA‑Implementierung. Für weitergehende
AutoML‑Funktionalität kann optional `hyperts` eingesetzt werden. Ist diese
Bibliothek installiert, wird sie automatisch verwendet, andernfalls
fällt der Code auf ARIMA zurück.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd

try:
    # HyperTS ist optional; falls verfügbar, verwenden wir es bevorzugt
    from hyperts import make_experiment  # type: ignore
    _HYPERTS_AVAILABLE = True
except ImportError:
    _HYPERTS_AVAILABLE = False

from sklearn.metrics import mean_squared_error

import statsmodels.api as sm


def run_forecasting(
    file_path: str,
    index_col: str,
    value_col: str,
    test_size: float = 0.2,
    arima_order: tuple[int, int, int] = (1, 1, 0),
) -> Tuple[pd.Series, pd.Series, float]:
    """Führt ein Forecasting auf einer univariaten Zeitreihe durch.

    Args:
        file_path: Pfad zur CSV‑Datei, die Datum/Uhrzeit und den Wert enthält.
        index_col: Name der Datums‑/Zeitspalte, die als Index genutzt wird.
        value_col: Name der Spalte mit den zu prognostizierenden Werten.
        test_size: Anteil des Datensatzes, der für das Testset verwendet wird.
        arima_order: Parameter (p, d, q) für das ARIMA‑Modell im Fallback.

    Returns:
        Tuple aus (true_values, predictions, mse):
        - true_values: die tatsächlichen Testwerte der Zeitreihe
        - predictions: die vorhergesagten Werte
        - mse: der mittlere quadratische Fehler

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
        ValueError: Wenn index_col oder value_col nicht existieren.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei {file_path} wurde nicht gefunden.")
    df = pd.read_csv(file_path, parse_dates=[index_col])
    if index_col not in df.columns or value_col not in df.columns:
        raise ValueError(
            f"Die Spalten {index_col} oder {value_col} wurden nicht in der CSV gefunden."
        )
    df = df.sort_values(by=index_col).set_index(index_col)
    series = df[value_col].astype(float)
    n = len(series)
    split_idx = int(n * (1 - test_size))
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    # Falls HyperTS verfügbar ist, verwenden wir dessen AutoML
    if _HYPERTS_AVAILABLE:
        # HyperTS erfordert ein DataFrame mit 'ds' und 'y' Spalten
        ts_df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })
        train_df = ts_df.iloc[:split_idx]
        test_df = ts_df.iloc[split_idx:]
        exp = make_experiment(
            train_df,
            task='univariate-forecast',
            timestamp='ds',
            target='y',
            eval_method='holdout',
            test_size=len(test_df)
        )
        model = exp.run()
        # Für Zeitreihen muss der Forecast‑Horizont explicit angegeben werden
        preds = model.predict(test_df)
        # Die Predictions sind evtl. als DataFrame zurückgegeben
        if isinstance(preds, pd.DataFrame):
            preds_series = preds['yhat'] if 'yhat' in preds.columns else preds.iloc[:, 0]
        else:
            preds_series = pd.Series(preds)
        mse = mean_squared_error(test_series.values, preds_series.values)
        return test_series, preds_series, mse
    else:
        # ARIMA Fallback
        model = sm.tsa.ARIMA(train_series, order=arima_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test_series))
        preds_series = pd.Series(forecast, index=test_series.index)
        mse = mean_squared_error(test_series.values, preds_series.values)
        return test_series, preds_series, mse