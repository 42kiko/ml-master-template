"""Klassifikationspipeline für tabellarische Daten.

Diese Funktion übernimmt den kompletten Ablauf: Laden einer CSV‑Datei,
Aufteilen in Trainings‑ und Testdaten, Trainieren eines Klassifikationsmodells
und Ausgeben der wichtigsten Metriken. Standardmäßig wird ein RandomForest
verwendet, da dieser gut mit heterogenen Featuretypen umgehen kann.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


def run_classification(
    file_path: str,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> Tuple[float, str]:
    """Führt eine Klassifikation auf dem angegebenen Datensatz aus.

    Args:
        file_path: Pfad zur CSV‑Datei mit den Daten. Die Datei sollte eine
            Spalte enthalten, die das Ziel (Label) repräsentiert.
        target_col: Name der Zielspalte.
        test_size: Anteil der Daten, der als Testset verwendet wird.
        random_state: Zufallsseed für reproduzierbare Ergebnisse.
        n_estimators: Anzahl der Bäume im RandomForest.
        max_depth: Maximale Tiefe der Bäume (optional).

    Returns:
        Tuple aus der Genauigkeit und dem Klassifikationsreport als Text.

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

    # One‑hot‑encoding für kategorische Variablen wird automatisch von Pandas get_dummies übernommen
    X_processed = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, max_depth=max_depth
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report