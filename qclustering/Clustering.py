from qclustering.KernelKMeans import KernelKMeans
from qclustering.OriginalKernelKMeans import OriginalKernelKMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.svm import SVC
from pennylane import numpy as np

def clustering(algorithm, algorithm_params, kernel, n_clusters, X):
    if callable(kernel):
        # kernel parameter is a function
        #distance = lambda x1, x2: 1 - kernel(x1, x2)
        distance = lambda x1, x2: np.sqrt(2 * (1 - np.sqrt(kernel(x1, x2))))
        X_distance = X
    elif kernel == "precomputed":
        # kernel parameter is "precomputed" and X is the kernel matrix
        distance = "precomputed"
        #X_distance = 1 - X
        X_distance = np.sqrt(2 * (1 - np.sqrt(X)))
        X_distance = X_distance.clip(min=0)

    if algorithm  == "kmeans":
        kmeans = KernelKMeans(n_clusters=n_clusters, kernel=kernel, **algorithm_params)
        return kmeans.fit(X).labels_
    elif algorithm == "original-kmeans":
        kmeans = OriginalKernelKMeans(n_clusters=n_clusters, kernel=kernel, **algorithm_params)
        return kmeans.fit(X).labels_
    elif algorithm == "spectral":
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=kernel, **algorithm_params)
        return spectral.fit(X).labels_
    elif algorithm == "agglomerative":
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=distance, **algorithm_params)
        return agglomerative.fit(X_distance).labels_
    elif algorithm == "dbscan":
        dbscan = DBSCAN(metric=distance, **algorithm_params)
        return dbscan.fit(X_distance).labels_
    elif algorithm == "optics":
        optics = OPTICS(metric=distance, **algorithm_params)
        return optics.fit(X_distance).labels_
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

def classification(train_X, train_Y, X, kernel, algorithm_params):
    svm = SVC(kernel=kernel, **algorithm_params).fit(train_X, train_Y)
    return svm.predict(X)