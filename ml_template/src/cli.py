import os
from src.utils.config_handler import load_config
from src.utils.logger import get_logger
from src.data_loader import load_dataset
from src.modeling.training import run_supervised_experiment, run_unsupervised_experiment, run_timeseries_experiment

logger = get_logger(__name__)

def prompt_menu(title, options):
    print(f"\n=== {title} ===")
    for i, opt in enumerate(options, 1):
        print(f"{i} - {opt}")
    while True:
        choice = input("> ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return int(choice)
        print("Bitte eine gültige Zahl eingeben.")

def run_cli():
    cfg = load_config("config/config.yaml")
    print("Willkommen zum ML-Template!")
    top = prompt_menu("Was möchtest du tun?", ["Supervised Learning", "Unsupervised Learning", "Time Series Forecasting"])
    if top == 1:
        goal = prompt_menu("Ziel?", ["Regression", "Classification"])
        problem_type = "regression" if goal == 1 else "classification"
        df = load_dataset(cfg)
        run_supervised_experiment(df, cfg, problem_type)
    elif top == 2:
        df = load_dataset(cfg, require_target=False)
        run_unsupervised_experiment(df, cfg)
    else:
        df = load_dataset(cfg, timeseries=True)
        run_timeseries_experiment(df, cfg)
    print("\nFertig! Ergebnisse unter reports/.")
