import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.preprocessing.timeseries_preprocessing import TimeSeriesFeatureGenerator

class SARIMAXWrapper:
    def __init__(self, order=(1,0,0), seasonal_order=(0,0,0,0)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None
        self.res_ = None

    def fit(self, X, y):
        endog = np.asarray(y)
        self.model_ = SARIMAX(endog, order=self.order, seasonal_order=self.seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        self.res_ = self.model_.fit(disp=False)
        return self

    def predict(self, X):
        # one-step ahead predictions length of X
        return self.res_.get_prediction(start=self.res_.nobs, end=self.res_.nobs + len(X) - 1).predicted_mean

def get_timeseries_models(datetime_col, target_col, cfg_ts):
    feat = TimeSeriesFeatureGenerator(
        datetime_col=datetime_col,
        target_col=target_col,
        lags=cfg_ts.get("lags", [1,7,30]),
        rolling_windows=cfg_ts.get("rolling_windows", [7,30]),
        add_fourier=cfg_ts.get("add_fourier", True),
        fourier_periods=cfg_ts.get("fourier_periods", [7,30]),
        seasonal_decompose_flag=cfg_ts.get("seasonal_decompose", False),
    )
    xgb_pipe = Pipeline([("feat", feat),
                         ("xgb", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42))])
    # For SARIMAX we handle separately (no sklearn pipeline)
    return {"xgb_lag": xgb_pipe, "sarimax": SARIMAXWrapper()}

def timeseries_cv_forecast(df, datetime_col, target_col, model, n_splits=5):
    df = df.sort_values(datetime_col).reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds = []
    trues = []
    for tr_idx, te_idx in tscv.split(df):
        train, test = df.iloc[tr_idx], df.iloc[te_idx]
        if hasattr(model, "fit") and hasattr(model, "predict"):
            if model.__class__.__name__ == "SARIMAXWrapper":
                y_tr = train[target_col].values
                model.fit(None, y_tr)
                y_hat = model.predict(test)
            else:
                X_tr = train.copy()
                y_tr = train[target_col].values
                X_te = test.copy()
                model.fit(X_tr, y_tr)
                y_hat = model.predict(X_te)
            preds.extend(np.asarray(y_hat).ravel().tolist())
            trues.extend(test[target_col].values.tolist())
    return np.array(trues), np.array(preds)
