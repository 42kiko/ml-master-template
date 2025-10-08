import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import seasonal_decompose

class TimeSeriesFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col: str, target_col: str, lags=None, rolling_windows=None,
                 add_fourier=True, fourier_periods=None, seasonal_decompose_flag=False):
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.lags = lags or [1,7,30]
        self.rolling_windows = rolling_windows or [7,30]
        self.add_fourier = add_fourier
        self.fourier_periods = fourier_periods or [7,30]
        self.seasonal_decompose_flag = seasonal_decompose_flag
        self.cols_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if self.datetime_col not in df.columns:
            raise ValueError(f"'{self.datetime_col}' not in columns")
        if self.target_col not in df.columns:
            raise ValueError(f"'{self.target_col}' not in columns")

        df = df.sort_values(self.datetime_col).reset_index(drop=True)

        # Datetime features
        dt = pd.to_datetime(df[self.datetime_col])
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["weekday"] = dt.dt.weekday
        df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
        df["quarter"] = dt.dt.quarter
        df["is_weekend"] = (df["weekday"] >= 5).astype(int)

        # Lags
        for lag in self.lags:
            df[f"lag_{lag}"] = df[self.target_col].shift(lag)

        # Rolling stats
        for w in self.rolling_windows:
            df[f"roll_mean_{w}"] = df[self.target_col].rolling(w).mean()
            df[f"roll_std_{w}"] = df[self.target_col].rolling(w).std()

        # Fourier terms
        if self.add_fourier:
            t = np.arange(len(df))
            for p in self.fourier_periods:
                df[f"fourier_sin_{p}"] = np.sin(2*np.pi*t/p)
                df[f"fourier_cos_{p}"] = np.cos(2*np.pi*t/p)

        # Seasonal decompose (optional)
        if self.seasonal_decompose_flag and len(df) > 2*max(self.lags+[7]):
            try:
                res = seasonal_decompose(df[self.target_col], period=max(self.lags+[7]), model="additive", extrapolate_trend="freq")
                df["trend"] = res.trend
                df["seasonal"] = res.seasonal
                df["resid"] = res.resid
            except Exception:
                # Fallback: ignore if decomposition fails
                pass

        # Drop rows with NaNs created by lag/rolling
        df = df.dropna().reset_index(drop=True)
        self.cols_ = [c for c in df.columns if c not in [self.datetime_col, self.target_col]]
        return df[self.cols_]
