from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def build_tabular_preprocessor(df: pd.DataFrame, target: str):
    numeric = df.drop(columns=[target]).select_dtypes(include=["number"]).columns.tolist()
    categorical = df.drop(columns=[target]).select_dtypes(exclude=["number"]).columns.tolist()
    num_pipe = make_numeric_pipeline()
    cat_pipe = make_categorical_pipeline()
    pre = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ], remainder="drop")
    return pre, numeric, categorical

def make_numeric_pipeline():
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def make_categorical_pipeline():
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
