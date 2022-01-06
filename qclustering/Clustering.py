from qclustering.KernelKMeans import KernelKMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS

def clustering(algorithm, kernel_obj, kernel_params, X, y=None, **algorithm_params):
    if algorithm  == "kmeans":
        kmeans = KernelKMeans(kernel=lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, kernel_params), **algorithm_params)
        return kmeans.fit(X, y).labels_
    elif algorithm == "spectral":
        spectral = SpectralClustering(affinity=lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, kernel_params), **algorithm_params)
        return spectral.fit(X, y).labels_
    elif algorithm == "agglomerative":
        agglomerative = AgglomerativeClustering(affinity=lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, kernel_params), **algorithm_params)
        return agglomerative.fit(X, y).labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric=lambda x1, x2: kernel_obj.distance_with_params(x1, x2, kernel_params), **algorithm_params)
        return dbscan.fit(X, y).labels_
    elif algorithm == "optics":
        optics = OPTICS(metric=lambda x1, x2: kernel_obj.distance_with_params(x1, x2, kernel_params), **algorithm_params)
        return optics.fit(X, y).labels_
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")