"""Regressionspipeline für tabellarische Daten.

Diese Funktion trainiert ein RandomForestRegressor‑Modell auf dem bereitgestellten
CSV‑Datensatz und berechnet Fehlerkenngrößen wie das mittlere quadratische
Fehlermaß (MSE) und den Bestimmtheitsmaß (R^2). Kategorische Variablen
werden per One‑Hot‑Encoding verarbeitet.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


def run_regression(
    file_path: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> Tuple[float, float]:
    """Führt eine Regression auf dem angegebenen Datensatz aus.

    Args:
        file_path: Pfad zur CSV‑Datei mit den Daten.
        target_col: Name der Zielspalte (numerisch).
        test_size: Anteil der Daten, der als Testset verwendet wird.
        random_state: Zufallsseed für reproduzierbare Ergebnisse.
        n_estimators: Anzahl der Bäume im RandomForest.
        max_depth: Maximale Tiefe der Bäume (optional).

    Returns:
        Tuple aus (mse, r2), wobei mse der mittlere quadratische Fehler und r2 der
        Bestimmtheitsmaß ist.

    Raises:
        FileNotFoundError: Wenn die CSV‑Datei nicht existiert.
        ValueError: Wenn die Zielspalte nicht im DataFrame enthalten ist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei {file_path} wurde nicht gefunden.")
    df = pd.read_csv(file_path)
    if target_col not in df.columns:
        raise ValueError(
            f"Die Zielspalte '{target_col}' wurde nicht in den Spalten gefunden: {list(df.columns)}"
        )
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_processed = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )
    reg = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state, max_depth=max_depth
    )
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2