import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_dataset(cfg, require_target=True, timeseries=False):
    path = cfg["data"]["input_csv"]
    target_col = cfg["data"]["target_column"]
    dt_col = cfg["data"]["datetime_column"]
    df = pd.read_csv(path)
    logger.info(f"Loaded data: {path}, shape={df.shape}")
    if timeseries:
        if dt_col not in df.columns:
            raise ValueError(f"datetime_column '{dt_col}' nicht in Daten gefunden")
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.sort_values(dt_col).reset_index(drop=True)
    if require_target and target_col not in df.columns:
        raise ValueError(f"target_column '{target_col}' nicht in Daten gefunden")
    return df
