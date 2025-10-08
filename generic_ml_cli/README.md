# Generic ML CLI Project

Dieses Projekt bietet ein generisches Kommandozeilen‑Werkzeug für Machine‑Learning‑Aufgaben und vereint mehrere Open‑Source‑Frameworks, die im Laufe der Recherche identifiziert wurden. Es richtet sich an Anwender:innen, die schnell zwischen unterschiedlichen Aufgaben (Supervised, Unsupervised und Zeitreihen) wechseln möchten und dabei von bereits existierenden Templates und Frameworks profitieren wollen.

## Hintergrund und Motivation

Die Bibliotheken **sktime** und **aeon** verfolgen das Ziel, eine einheitliche Schnittstelle für verschiedene Zeitreihen‑Lernaufgaben bereitzustellen. Laut der Dokumentation unterstützt `sktime` aktuell Zeitreihen‑**Klassifikation**, **Regression**, **Clustering** und **Forecasting**, wobei jede Aufgabe über ein gemeinsames API abgewickelt werden kann【744458527013480†L101-L122】. Das `aeon`‑Projekt ergänzt dieses Konzept und hebt hervor, dass es ein scikit‑learn‑kompatibles Toolkit für Aufgaben wie **Klassifikation**, **Regression**, **Clustering**, **Anomalieerkennung**, **Segmentierung** und **Ähnlichkeitssuche** ist【708474877637852†L1262-L1273】.

Parallel dazu existiert mit **HyperTS** eine spezialisierte AutoML‑Bibliothek für Zeitreihen. Sie deckt den gesamten Workflow ab – von der Datenbereinigung über Feature‑Engineering und Modellauswahl bis zur Hyperparameter‑Optimierung und Visualisierung【379375404820139†L110-L133】. Ein wesentliches Merkmal von HyperTS ist die Unterstützung mehrerer Aufgaben: **Forecasting**, **Klassifikation**, **Regression** und **Anomalieerkennung**【379375404820139†L118-L125】. Diese Vielfalt macht HyperTS zu einem wertvollen Baustein für ein allumfassendes Projekt.

Ziel dieses Projekts ist es, diese Erkenntnisse zu einem generischen, erweiterbaren CLI‑Tool zu vereinen. Anwender:innen sollen per Terminal abgefragt werden, welche Aufgabe durchgeführt werden soll, woraufhin das Programm automatisch die passende Pipeline aufbaut – von der Datenaufbereitung über den Train‑Test‑Split bis hin zur Auswertung der Ergebnisse.

## Funktionen

- **Zeitreihen‑Forecasting** – Umsetzung mit `statsmodels` (ARIMA) für einfache Vorhersagen. Optional lässt sich HyperTS integrieren: es ermöglicht eine einheitliche Oberfläche für Forecasting, Klassifikation, Regression und Anomalieerkennung【379375404820139†L118-L125】.
- **Zeitreihen‑Klassifikation/Regression/Clustering** – durch die Integration von `sktime` oder `aeon` können diese Aufgaben mit einem gemeinsamen API ausgeführt werden. Falls die Bibliotheken nicht installiert sind, werden einfache Fallback‑Modelle genutzt.
- **Supervised Learning (Tabellarische Daten)** – Klassifikation und Regression mittels scikit‑learn. Das Tool übernimmt automatisch das Einlesen von CSV‑Dateien, den Train‑Test‑Split und die Berechnung gängiger Metriken.
- **Unsupervised Learning / Clustering** – Einsatz von scikit‑learn‑K‑Means für die Clusteranalyse; die Anzahl der Cluster wird interaktiv abgefragt.
- **Interaktive CLI** – nach dem Start fragt das Programm per Terminal, welche Aufgabe gewünscht ist, und fordert anschließend die benötigten Parameter (z. B. Pfad zur CSV, Zielspalte, Datumsspalte) ab.
- **Modulare Struktur** – Die Implementierung ist in einzelne Aufgabenmodule unterteilt. Dadurch lassen sich später weitere Methoden (z. B. die Einbindung von Aeon‑Algorithmen) leicht ergänzen.

## Installation

1. Repository klonen oder den Code herunterladen.
2. Optional: virtuelles Environment anlegen

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Abhängigkeiten installieren:

   ```bash
   pip install -r requirements.txt
   ```

   Um Zeitreihen‑Frameworks wie `sktime`, `aeon` oder `hyperts` zu nutzen, lassen sich diese optional nachinstallieren:

   ```bash
   pip install sktime aeon hyperts
   ```

## Nutzung

Das CLI‑Tool wird über das Python‑Modul ausgeführt:

```bash
python -m generic_ml_cli.src.cli
```

Anschließend erscheint ein Menü, in dem eine Aufgabe gewählt werden kann (z. B. Zeitreihen‑Forecasting, Klassifikation, Regression, Clustering). Das Programm fragt interaktiv nach den relevanten Eingaben (Pfad zur CSV‑Datei, Spaltennamen, Anzahl Cluster etc.) und führt die Pipeline aus. Ergebnisse wie Fehlermetriken oder Clusterzugehörigkeiten werden direkt im Terminal ausgegeben.

## Anforderungen

- Python ≥ 3.8
- In `requirements.txt` aufgeführte Pakete (pandas, scikit‑learn, statsmodels, numpy)
- Optionale Pakete für erweiterte Zeitreihen‑Funktionalität: `sktime`, `aeon`, `hyperts`

## Erweiterbarkeit

Die modulare Struktur (siehe Ordner `src/tasks`) erlaubt es, weitere Methoden oder Frameworks einzubinden. So ließen sich z. B. die in `aeon` verfügbaren state‑of‑the‑art‑Algorithmen für Zeitreihen‑Klassifikation oder ‑Clustering ergänzen. Da `aeon` scikit‑learn‑kompatibel ist, fügt es sich nahtlos in bestehende Pipelines ein【708474877637852†L1262-L1273】.

Für komplexe AutoML‑Workflows mit Hyperparameter‑Optimierung und Visualisierung empfiehlt sich zudem der Einsatz von **HyperTS**. Diese Bibliothek deckt laut Dokumentation die vollständige ML‑Pipeline ab – von der Datenvorbereitung über Feature‑Engineering bis hin zu Ergebnis‑Evaluation und Visualisierung【379375404820139†L110-L133】.
