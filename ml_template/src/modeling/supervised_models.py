from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

def get_supervised_models(problem_type: str, preprocessor):
    if problem_type == "regression":
        models = {
            "linreg": Pipeline([("pre", preprocessor), ("model", LinearRegression())]),
            "ridge": Pipeline([("pre", preprocessor), ("model", Ridge())]),
            "rf": Pipeline([("pre", preprocessor), ("model", RandomForestRegressor(n_estimators=300, random_state=42))]),
            "xgb": Pipeline([("pre", preprocessor), ("model", XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42))]),
        }
        param_grids = {
            "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "rf": {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20]},
            "xgb": {"model__n_estimators": [300, 500], "model__max_depth": [4,6,8], "model__learning_rate":[0.03,0.05,0.1]},
        }
    else:
        models = {
            "logreg": Pipeline([("pre", preprocessor), ("model", LogisticRegression(max_iter=200))]),
            "rf": Pipeline([("pre", preprocessor), ("model", RandomForestClassifier(n_estimators=300, random_state=42))]),
            "xgb": Pipeline([("pre", preprocessor), ("model", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric="logloss"))]),
        }
        param_grids = {
            "logreg": {"model__C": [0.1, 1.0, 3.0]},
            "rf": {"model__n_estimators": [200, 400], "model__max_depth": [None, 10, 20]},
            "xgb": {"model__n_estimators": [300, 500], "model__max_depth": [4,6,8], "model__learning_rate":[0.03,0.05,0.1]},
        }
    return models, param_grids
