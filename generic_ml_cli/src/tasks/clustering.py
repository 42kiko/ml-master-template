"""Unüberwachtes Clustering mittels K‑Means.

Dieses Modul implementiert eine einfache K‑Means‑Clusteranalyse für
tabellarische Daten. Kategorische Features werden automatisch per One‑Hot‑
Encoding in numerische Repräsentationen überführt. Die Anzahl der Cluster
wird vom Benutzer angegeben.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_clustering(
    file_path: str,
    n_clusters: int,
    drop_cols: Optional[list[str]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, list[int]]:
    """Führt K‑Means‑Clustering auf dem angegebenen Datensatz aus.

    Args:
        file_path: Pfad zur CSV‑Datei mit den Daten.
        n_clusters: Anzahl der zu bildenden Cluster.
        drop_cols: Optionale Liste von Spalten, die vor dem Clustering entfernt
            werden sollen (z. B. Zielspalten oder IDs).
        random_state: Zufallsseed für reproduzierbare Ergebnisse.

    Returns:
        Ein Tupel aus (df_with_labels, labels_list), wobei
        - df_with_labels: das ursprüngliche DataFrame mit einer neuen Spalte
          `cluster`, die die Clusterzugehörigkeit enthält.
        - labels_list: die Liste der Clusterlabels in der Reihenfolge der
          Datenpunkte.

    Raises:
        FileNotFoundError: Wenn die CSV‑Datei nicht existiert.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Die Datei {file_path} wurde nicht gefunden.")
    df = pd.read_csv(file_path)
    if drop_cols:
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
    # One‑Hot‑Encoding für kategorische Variablen
    df_encoded = pd.get_dummies(df)
    # Skalieren der Features für bessere K‑Means‑Performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_encoded)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)
    result = df.copy()
    result['cluster'] = labels
    return result, labels.tolist()