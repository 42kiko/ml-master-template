"""Kommandozeilenoberfläche für das generische ML‑Projekt.

Dieses Skript stellt eine interaktive CLI bereit, über die Nutzende
verschiedene Machine‑Learning‑Aufgaben ausführen können. Es fragt nach dem
gewünschten Aufgabenbereich, sammelt die nötigen Informationen und ruft die
entsprechenden Funktionen aus dem `tasks`‑Paket auf. Die Ergebnisse werden
übersichtlich im Terminal ausgegeben.
"""

from __future__ import annotations

import sys
from typing import Optional

try:
    import readline  # Optional: verbessert die Eingabeinteraktion
except ImportError:
    readline = None  # type: ignore

from .tasks import classification, regression, clustering, time_series


def prompt(msg: str) -> str:
    """Kapselt input() für eine bessere Testbarkeit.

    Args:
        msg: Die anzuzeigende Meldung.

    Returns:
        Die vom Benutzer eingegebene Zeichenkette.
    """
    return input(msg)


def menu() -> None:
    """Zeigt das Hauptmenü an."""
    print("\nBitte wählen Sie eine Aufgabe:")
    print("1) Zeitreihen‑Forecasting")
    print("2) Zeitreihen‑Klassifikation (erfordert sktime/aeon)")
    print("3) Zeitreihen‑Regression (erfordert sktime/aeon)")
    print("4) Zeitreihen‑Clustering (erfordert sktime/aeon)")
    print("5) Supervised Klassifikation (tabellarisch)")
    print("6) Supervised Regression (tabellarisch)")
    print("7) Unsupervised Clustering (tabellarisch)")
    print("q) Beenden")


def run_time_series_classification() -> None:
    """Placeholder für Zeitreihen‑Klassifikation mit sktime/aeon.

    Da weder sktime noch aeon standardmäßig installiert sind, informiert
    diese Funktion den Benutzer über die Notwendigkeit der Installation.
    """
    print(
        "Zeitreihen‑Klassifikation ist verfügbar, sobald Sie die Bibliotheken "
        "`sktime` oder `aeon` installieren. Bitte installieren Sie eines dieser "
        "Pakete und erweitern Sie das Script `tasks/time_series.py`, um entsprechende "
        "Modelle zu nutzen."
    )


def run_time_series_regression() -> None:
    """Placeholder für Zeitreihen‑Regression mit sktime/aeon."""
    print(
        "Zeitreihen‑Regression ist verfügbar, sobald Sie die Bibliotheken "
        "`sktime` oder `aeon` installieren. Bitte erweitern Sie das Projekt entsprechend."
    )


def run_time_series_clustering() -> None:
    """Placeholder für Zeitreihen‑Clustering mit sktime/aeon."""
    print(
        "Zeitreihen‑Clustering ist verfügbar, sobald Sie die Bibliotheken "
        "`sktime` oder `aeon` installieren. Bitte erweitern Sie das Projekt entsprechend."
    )


def main() -> None:
    """Einstiegspunkt der CLI."""
    print("Willkommen beim Generic ML CLI!")
    while True:
        menu()
        choice = prompt("Ihre Wahl: ").strip().lower()
        if choice == '1':
            try:
                file_path = prompt("Pfad zur CSV‑Datei mit Zeitreihendaten: ")
                index_col = prompt("Name der Datums‑/Zeitspalte: ")
                value_col = prompt("Name der Wertespalte: ")
                test_size_input = prompt("Testanteil (z. B. 0.2 für 20%): ") or "0.2"
                test_size = float(test_size_input)
                ts, preds, mse = time_series.run_forecasting(
                    file_path=file_path,
                    index_col=index_col,
                    value_col=value_col,
                    test_size=test_size,
                )
                print(f"\nErgebnis des Forecastings (MSE): {mse:.4f}")
                print("Erste 5 tatsächliche vs. prognostizierte Werte:")
                for t, p in zip(ts.head(5).items(), preds.head(5).items()):
                    # t und p sind Tupel (Index, Wert)
                    print(f"{t[0]}: wahr={t[1]:.4f}, prognose={p[1]:.4f}")
            except Exception as e:
                print(f"Fehler beim Forecasting: {e}")
        elif choice == '2':
            run_time_series_classification()
        elif choice == '3':
            run_time_series_regression()
        elif choice == '4':
            run_time_series_clustering()
        elif choice == '5':
            try:
                file_path = prompt("Pfad zur CSV‑Datei: ")
                target_col = prompt("Name der Zielspalte: ")
                test_size_input = prompt("Testanteil (z. B. 0.2 für 20%): ") or "0.2"
                test_size = float(test_size_input)
                acc, report = classification.run_classification(
                    file_path=file_path,
                    target_col=target_col,
                    test_size=test_size,
                )
                print(f"\nGenauigkeit: {acc:.4f}\n")
                print("Klassifikationsbericht:")
                print(report)
            except Exception as e:
                print(f"Fehler bei der Klassifikation: {e}")
        elif choice == '6':
            try:
                file_path = prompt("Pfad zur CSV‑Datei: ")
                target_col = prompt("Name der Zielspalte: ")
                test_size_input = prompt("Testanteil (z. B. 0.2 für 20%): ") or "0.2"
                test_size = float(test_size_input)
                mse, r2 = regression.run_regression(
                    file_path=file_path,
                    target_col=target_col,
                    test_size=test_size,
                )
                print(f"\nMittlerer quadratischer Fehler (MSE): {mse:.4f}")
                print(f"Bestimmtheitsmaß (R^2): {r2:.4f}")
            except Exception as e:
                print(f"Fehler bei der Regression: {e}")
        elif choice == '7':
            try:
                file_path = prompt("Pfad zur CSV‑Datei: ")
                n_clusters_input = prompt("Anzahl der Cluster: ")
                n_clusters = int(n_clusters_input)
                drop_cols_input = prompt(
                    "Durch Komma getrennte Spalten, die vor dem Clustering entfernt werden sollen (optional): "
                ).strip()
                drop_cols_list: Optional[list[str]] = None
                if drop_cols_input:
                    drop_cols_list = [c.strip() for c in drop_cols_input.split(',') if c.strip()]
                df_with_labels, labels = clustering.run_clustering(
                    file_path=file_path,
                    n_clusters=n_clusters,
                    drop_cols=drop_cols_list,
                )
                print("\nClustering abgeschlossen. Die ersten 5 Datensätze mit Clusterzuordnung:")
                print(df_with_labels.head())
            except Exception as e:
                print(f"Fehler beim Clustering: {e}")
        elif choice == 'q' or choice == 'quit':
            print("Auf Wiedersehen!")
            sys.exit(0)
        else:
            print("Ungültige Eingabe. Bitte erneut versuchen.")


if __name__ == "__main__":
    main()