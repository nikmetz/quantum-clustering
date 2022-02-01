from qclustering.KernelKMeans import KernelKMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS

def clustering(algorithm, algorithm_params, kernel_obj, n_clusters, X, train_X=None, train_Y=None):
    if algorithm  == "kmeans":
        kmeans = KernelKMeans(n_clusters=n_clusters, kernel=kernel_obj.kernel, **algorithm_params)
        return kmeans.fit(X).labels_
    elif algorithm == "spectral":
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=kernel_obj.kernel, **algorithm_params)
        return spectral.fit(X).labels_
    elif algorithm == "agglomerative":
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=kernel_obj.kernel, **algorithm_params)
        return agglomerative.fit(X).labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric=kernel_obj.distance, **algorithm_params)
        return dbscan.fit(X).labels_
    elif algorithm == "optics":
        optics = OPTICS(metric=kernel_obj.distance, **algorithm_params)
        return optics.fit(X).labels_
    elif algorithm == "svm":
        pass
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")