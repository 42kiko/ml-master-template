from sklearn.metrics import silhouette_score
import numpy as np

def evaluate_unsupervised(pipeline, X):
    try:
        step_name, step = pipeline.steps[-1]
        labels = pipeline.fit_predict(X) if step_name != "kmeans" else pipeline.predict(X)
        sil = silhouette_score(X, labels)
        return {"silhouette": sil}
    except Exception:
        return {"silhouette": float("nan")}
