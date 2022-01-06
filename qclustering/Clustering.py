from qclustering.KernelKMeans import KernelKMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS

def clustering(algorithm, kernel_obj, kernel_params, X, n_clusters, algorithm_params):
    if algorithm  == "kmeans":
        kmeans = KernelKMeans(n_clusters=n_clusters, kernel=lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, kernel_params), **algorithm_params)
        return kmeans.fit(X).labels_
    elif algorithm == "spectral":
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, kernel_params), **algorithm_params)
        return spectral.fit(X).labels_
    elif algorithm == "agglomerative":
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, kernel_params), **algorithm_params)
        return agglomerative.fit(X).labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric=lambda x1, x2: kernel_obj.distance_with_params(x1, x2, kernel_params), **algorithm_params)
        return dbscan.fit(X).labels_
    elif algorithm == "optics":
        optics = OPTICS(metric=lambda x1, x2: kernel_obj.distance_with_params(x1, x2, kernel_params), **algorithm_params)
        return optics.fit(X).labels_
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")