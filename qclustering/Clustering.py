from qclustering.KernelKMeans import KernelKMeans
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS
from sklearn.svm import SVC

def clustering(algorithm, algorithm_params, kernel, n_clusters, X):
    if callable(kernel):
        distance = lambda x1, x2: 1 - kernel(x1, x2)
        X_distance = X
    else:
        distance = kernel
        X_distance = 1 - X

    if algorithm  == "kmeans":
        kmeans = KernelKMeans(n_clusters=n_clusters, kernel=kernel, **algorithm_params)
        return kmeans.fit(X).labels_
    elif algorithm == "spectral":
        spectral = SpectralClustering(n_clusters=n_clusters, affinity=kernel, **algorithm_params)
        return spectral.fit(X).labels_
    elif algorithm == "agglomerative":
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity=kernel, **algorithm_params)
        return agglomerative.fit(X).labels_
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