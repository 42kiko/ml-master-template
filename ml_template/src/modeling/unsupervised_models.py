from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

def get_unsupervised_models(n_components=2):
    models = {
        "kmeans": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_components)), ("kmeans", KMeans(n_clusters=3, n_init=10, random_state=42))]),
        "dbscan": Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=n_components)), ("dbscan", DBSCAN(eps=0.5, min_samples=5))]),
    }
    return models
