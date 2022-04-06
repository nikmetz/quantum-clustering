from typing import NamedTuple
from sklearn import metrics
from qclustering.utils import simple_plot
from pennylane import numpy as np
from itertools import permutations
from pathlib import Path
from qclustering.Clustering import clustering, classification
from qclustering.KernelMatrix import parallel_kernel_matrix, parallel_square_kernel_matrix, square_postprocessing
import copy
import pickle

class ClusterResult(NamedTuple):
    labels: np.ndarray
    rand_score: float
    adjusted_rand_score: float
    calinski_harabasz: float
    davies_bouldin: float
    normalized_mutual_information: float

def accuracy(true_labels, pred_labels):
    return 1 - np.count_nonzero(pred_labels - true_labels) / len(true_labels)

def cluster_accuracy(true_labels, pred_labels):
    acc = 0.0
    unique_labels = np.unique(true_labels)
    permut = set(permutations(unique_labels))
    unique_pred_labels = np.unique(pred_labels)

    for x in permut:
        a = copy.deepcopy(pred_labels)
        masks = []
        for l in unique_pred_labels:
            masks.append(a == l)
        for idx, y in enumerate(masks):
            a[y] = x[idx]

        acc = max(acc, accuracy(true_labels, a))
    return acc

def get_cluster_result(algorithm, X, labels_true, labels_pred) -> ClusterResult:
    #if algorithm == "svm":
    #    acc = accuracy(labels_true, labels_pred)
    #else:
    #    acc = cluster_accuracy(labels_true, labels_pred)
    rand_score = metrics.rand_score(labels_true, labels_pred)
    adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    normalized_mutual_information = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    if len(np.unique(labels_pred)) < 2:
        calinski_harabasz = 0
        davies_bouldin = 0
    else:
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels_pred)
        davies_bouldin = metrics.davies_bouldin_score(X, labels_pred)

    return ClusterResult(labels_pred, rand_score, adjusted_rand_score, calinski_harabasz, davies_bouldin, normalized_mutual_information)

class Logging:

    def __init__(self, data, test_clustering_interval, train_clustering_interval, testing_algorithms, testing_algorithm_params, path):
        self.test_clustering_interval = test_clustering_interval
        self.train_clustering_interval = train_clustering_interval
        self.testing_algorithms = testing_algorithms
        self.testing_algorithm_params = testing_algorithm_params
        self.path = path
        self.prepare_folders()

        self.data = data
        self.weights = {}
        self.train_cost = {}
        self.val_cost = {}
        self.test_clustering = {}
        self.test_kernel_matrix = {}
        self.train_clustering = {}
        self.train_kernel_matrix = {}

    def prepare_folders(self):
        for x in self.testing_algorithms:
            Path(self.path, "test-clustering", x).mkdir(parents=True, exist_ok=True)
            Path(self.path, "train-clustering", x).mkdir(parents=True, exist_ok=True)

    def log_train_cost(self, step: int, cost: float):
        self.train_cost[step] = cost

    def log_val_cost(self, step: int, val_cost: float):
        self.val_cost[step] = val_cost

    def log_weights(self, step: int, weights):
        self.weights[step] = weights
    
    def log_train_clustering(self, epoch, kernel, n_clusters, train_X, train_Y):
        if epoch % self.train_clustering_interval == 0:
            self.clustering_eval(epoch, kernel, n_clusters, self.train_clustering, self.train_kernel_matrix, train_X, train_Y, train_X=train_X, train_Y=train_Y)

    def log_test_clustering(self, epoch, kernel, n_clusters, test_X, test_Y, train_X=None, train_Y=None):
        if epoch % self.test_clustering_interval == 0:
            self.clustering_eval(epoch, kernel, n_clusters, self.test_clustering, self.test_kernel_matrix, test_X, test_Y, train_X=train_X, train_Y=train_Y)

    def clustering_eval(self, epoch, kernel, n_clusters, clustering_data, kernel_data, X, Y, train_X=None, train_Y=None):
        X_kernel = parallel_square_kernel_matrix(X, kernel)
        X_kernel = square_postprocessing(X_kernel)
        kernel_data[epoch] = X_kernel

        for idx, alg in enumerate(self.testing_algorithms):
            if alg == "svm":
                train_X_kernel = parallel_square_kernel_matrix(train_X, kernel)
                train_X_kernel = square_postprocessing(train_X_kernel)
                train_test_X_kernel = parallel_kernel_matrix(X, train_X, kernel)
                labels = classification(train_X_kernel, train_Y, train_test_X_kernel, "precomputed", self.testing_algorithm_params[idx])

            else:
                labels = clustering(alg, self.testing_algorithm_params[idx], "precomputed", n_clusters, X_kernel)
            result = get_cluster_result(alg, X, Y, labels)

            if not alg in clustering_data:
                clustering_data[alg] = {}
            clustering_data[alg][epoch] = result

    def finish(self):
        pickle_data = {
            "data": self.data,
            "weights": self.weights,
            "train_cost": self.train_cost,
            "val_cost": self.val_cost,
            "test_clustering": self.test_clustering,
            "test_kernel_matrix": self.test_kernel_matrix,
            "train_clustering": self.train_clustering,
            "train_kernel_matrix": self.train_kernel_matrix
        }
        pickle.dump(pickle_data, open(Path(self.path, "logging.p"), "wb"))

        train_steps = list(self.train_cost.keys())
        simple_plot(train_steps, [self.train_cost[x] for x in train_steps], "Steps", "Train loss", Path(self.path, "train_loss.png"))
        val_steps = list(self.val_cost.keys())
        simple_plot(val_steps, [self.val_cost[x] for x in val_steps], "Steps", "Validation loss", Path(self.path, "val_loss.png"))

        for alg, result in self.test_clustering.items():
            self.generate_cluster_imgs(Path(self.path, "test-clustering", alg), result)
        for alg, result in self.train_clustering.items():
            self.generate_cluster_imgs(Path(self.path, "train-clustering", alg), result)


    def generate_cluster_imgs(self, path, result):
        clustering_steps = list(result.keys())
        simple_plot(clustering_steps, [result[x].rand_score for x in clustering_steps], "Epoch", "Rand score", Path(path, "rand_score.png"), [0,1.1])
        simple_plot(clustering_steps, [result[x].adjusted_rand_score for x in clustering_steps], "Epoch", "Adjusted rand score", Path(path, "adjusted_rand_score.png"), [-1.1, 1.1])
        simple_plot(clustering_steps, [result[x].calinski_harabasz for x in clustering_steps], "Epoch", "Calinski harabasz", Path(path, "calinski_harabasz.png"))
        simple_plot(clustering_steps, [result[x].davies_bouldin for x in clustering_steps], "Epoch", "Davies bouldin", Path(path, "davies_bouldin.png"))
        simple_plot(clustering_steps, [result[x].normalized_mutual_information for x in clustering_steps], "Epoch", "Noralized mutual information", Path(path, "normalized_mutual_information.png"))