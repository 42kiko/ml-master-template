import os
import mlflow
from contextlib import contextmanager

def init_mlflow(experiment_name: str = "default_experiment"):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(experiment_name)

@contextmanager
def start_run(run_name: str = None, tags: dict = None):
    with mlflow.start_run(run_name=run_name, tags=tags):
        yield

def log_params(params: dict):
    if params:
        mlflow.log_params({k:str(v) for k,v in params.items()})

def log_metrics(metrics: dict, step: int = None):
    if metrics:
        mlflow.log_metrics({k: float(v) for k,v in metrics.items()}, step=step)

def log_artifact(path: str):
    if os.path.exists(path):
        mlflow.log_artifact(path)

def log_artifacts(path: str):
    if os.path.exists(path):
        mlflow.log_artifacts(path)
