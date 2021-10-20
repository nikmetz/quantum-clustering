import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.cluster import KMeans
import pennylane as qml
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding

n_qubits = 4

dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

def euclidean_dist(x, y):
    return np.linalg.norm(x - y)

@qml.qnode(dev_kernel)
def amplitude_kernel(x1, x2):
    AmplitudeEmbedding(x1, wires=range(n_qubits), normalize=True)
    qml.adjoint(AmplitudeEmbedding)(x2, wires=range(n_qubits), normalize=True)
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

@qml.qnode(dev_kernel)
def angle_kernel(x1, x2):
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

def kmeans(data, num_clusters, init, dist_func):
    num_samples = data.shape[0]
    sample_dim = data.shape[1]
    centroids = np.zeros(shape=(num_clusters, sample_dim))

    for i in range(num_clusters):
        rand = np.random.randint(num_samples)
        centroids[i] = data[rand]

    classification = []
    iterations = 5
    for i in range(iterations):
        classification = []
        for x in data:
            min_dist = None
            c = None
            for idx, k in enumerate(centroids):
                dist = dist_func(x, k)
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    c = idx
            classification.append(c)

        for k in range(num_clusters):
            fitting = []
            for idx, label in enumerate(classification):
                if label == k:
                    fitting.append(data[idx])
            centroids[k] = np.mean(np.asarray(fitting), axis=0)
    return classification



data, target = datasets.load_iris(return_X_y=True)

a = kmeans(data, 5, "random", angle_kernel)

print(metrics.rand_score(target, a))
print(metrics.adjusted_rand_score(target, a))

#k = KMeans(n_clusters=3, random_state=0).fit(data)
#print(metrics.rand_score(target, k.labels_))
#print(metrics.adjusted_rand_score(target, k.labels_))