from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
import numpy as np

def evaluate_supervised(estimator, X, y, problem_type):
    y_pred = estimator.predict(X)
    if problem_type == "classification":
        f1 = f1_score(y, y_pred, average="macro")
        score = f1
        metrics = {"f1_macro": f1, "score": score}
    else:
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        score = -rmse
        metrics = {"rmse": rmse, "r2": r2, "score": score}
    return y, y_pred, metrics
