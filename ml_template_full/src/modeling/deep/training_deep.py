import os, time, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from joblib import dump
from src.utils.logger import get_logger
from src.tracking.mlflow_utils import init_mlflow, start_run, log_params, log_metrics, log_artifacts
from src.modeling.deep.tabular_mlp import TabularMLP
from src.modeling.deep.ts_lstm import LSTMForecaster

logger = get_logger(__name__)

def _to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

def train_tabular_mlp(X, y, task="regression", epochs=20, batch_size=64, lr=1e-3):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_t = _to_tensor(Xs)
    if task == "classification" and len(np.unique(y)) > 2:
        out_features = len(np.unique(y))
        y_t = torch.tensor(y, dtype=torch.long)
    elif task == "classification":
        out_features = 1
        y_t = torch.tensor(y, dtype=torch.float32).view(-1,1)
    else:
        out_features = 1
        y_t = torch.tensor(y, dtype=torch.float32).view(-1,1)

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = TabularMLP(in_features=X_t.shape[1], out_features=out_features, task=task)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if task=="regression" else (nn.BCEWithLogitsLoss() if out_features==1 else nn.CrossEntropyLoss())

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            if task=="classification" and out_features==1:
                loss = criterion(out, yb)
            else:
                loss = criterion(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log_metrics({"train_loss": float(np.mean(losses))}, step=epoch)
    return model, scaler

def make_ts_windows(series, lookback=30, horizon=1):
    X, y = [], []
    arr = np.asarray(series).reshape(-1,1)
    for i in range(lookback, len(arr)-horizon+1):
        X.append(arr[i-lookback:i])
        y.append(arr[i+horizon-1])
    return np.array(X), np.array(y)

def train_lstm_forecaster(y_series, lookback=30, horizon=1, epochs=20, batch_size=64, lr=1e-3):
    X, y = make_ts_windows(y_series, lookback=lookback, horizon=horizon)
    ds = TensorDataset(_to_tensor(X), _to_tensor(y))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    opt = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log_metrics({"train_loss": float(np.mean(losses))}, step=epoch)
    return model

def run_deep_tabular(df, cfg, task="regression"):
    init_mlflow(cfg.get("experiment_name", "default_experiment"))
    with start_run(run_name=f"deep_tabular_{task}"):
        target = cfg["data"]["target_column"]
        X = df.drop(columns=[target]).select_dtypes(include=[np.number]).values
        y = df[target].values
        model, scaler = train_tabular_mlp(X, y, task=task, epochs=10, batch_size=64, lr=1e-3)
        os.makedirs("reports", exist_ok=True)
        dump((model.state_dict(), scaler), "reports/deep_tabular.joblib")
        log_artifacts("reports")

def run_deep_timeseries(df, cfg):
    init_mlflow(cfg.get("experiment_name", "default_experiment"))
    with start_run(run_name="deep_timeseries_lstm"):
        dt_col = cfg["data"]["datetime_column"]
        target_col = cfg["data"]["target_column"]
        y = df.sort_values(dt_col)[target_col].values
        model = train_lstm_forecaster(y, lookback=30, horizon=1, epochs=10, batch_size=64, lr=1e-3)
        import torch
        os.makedirs("reports", exist_ok=True)
        torch.save(model.state_dict(), "reports/ts_lstm.pt")
        log_artifacts("reports")
