from sklearn.metrics import silhouette_score
import numpy as np

def evaluate_unsupervised(pipeline, X):
    # Try to infer labels if KMeans, else fallback to NaN metrics
    try:
        # last step name could be 'kmeans' or 'dbscan'
        step_name, step = pipeline.steps[-1]
        if step_name == "kmeans":
            labels = pipeline.predict(X)
        else:
            labels = pipeline.fit_predict(X)
        sil = silhouette_score(X, labels)
        return {"silhouette": sil}
    except Exception:
        return {"silhouette": float("nan")}
