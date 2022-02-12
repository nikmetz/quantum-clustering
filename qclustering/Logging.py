from typing import NamedTuple
from sklearn import metrics
from qclustering.utils import simple_plot, plot_kernel
from pennylane import numpy as np
from itertools import permutations
from pathlib import Path
from qclustering.Clustering import clustering, classification
from qclustering.utils import parallel_kernel_matrix, parallel_square_kernel_matrix
import matplotlib.pyplot as plt
import copy
import csv
import pennylane as qml

CLUSTER_HEADING = ["epoch", "rand_score", "adjusted_rand_score", "calinksi_harabasz", "davies_bouldin"]

class ClusterResult(NamedTuple):
    rand_score: float
    adjusted_rand_score: float
    calinski_harabasz: float
    davies_bouldin: float

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
    if len(np.unique(labels_pred)) < 2:
        calinski_harabasz = 0
        davies_bouldin = 0
    else:
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels_pred)
        davies_bouldin = metrics.davies_bouldin_score(X, labels_pred)

    return ClusterResult(rand_score, adjusted_rand_score, calinski_harabasz, davies_bouldin)

class Logging:

    def __init__(self, testing_interval, testing_algorithms, testing_algorithm_params, process_pool, path):
        self.testing_interval = testing_interval
        self.testing_algorithms = testing_algorithms
        self.testing_algorithm_params = testing_algorithm_params
        self.process_pool = process_pool
        self.path = path
        self.prepare_folders()

        self.train_cost = {}
        self.val_cost = {}
        self.testing = {}

    def prepare_folders(self):
        Path(self.path, "kernel").mkdir(parents=True, exist_ok=True)
        for x in self.testing_algorithms:
            Path(self.path, x).mkdir(parents=True, exist_ok=True)

    def log_training(self, step: int, cost: float):
        self.train_cost[step] = cost

    def log_validation(self, step: int, val_cost: float):
        self.val_cost[step] = val_cost

    def log_testing(self, epoch, kernel, n_clusters, test_X, test_Y, train_X=None, train_Y=None):
        if epoch % self.testing_interval == 0:
            X_kernel = parallel_square_kernel_matrix(test_X, kernel, pool=self.process_pool)

            plot_kernel(X_kernel, test_Y, epoch, Path(self.path, "kernel", str(epoch)+".png"))

            for idx, alg in enumerate(self.testing_algorithms):
                if alg == "svm":
                    train_X_kernel = parallel_square_kernel_matrix(train_X, kernel, pool=self.process_pool)
                    train_test_X_kernel = parallel_kernel_matrix(test_X, train_X, kernel, pool=self.process_pool)
                    labels = classification(train_X_kernel, train_Y, train_test_X_kernel, "precomputed", self.testing_algorithm_params[idx])

                else:
                    labels = clustering(alg, self.testing_algorithm_params[idx], "precomputed", n_clusters, X_kernel)
                result = get_cluster_result(alg, test_X, test_Y, labels)

                if not alg in self.testing:
                    self.testing[alg] = {}
                self.testing[alg][epoch] = result

                if test_X.shape[1] == 2:
                    p = Path(self.path, alg, "visualization")
                    p.mkdir(parents=True, exist_ok=True)
                    self.visualize_cluster(test_X, labels, Path(p, str(epoch)+".png"))

    def write_csv(self, path, heading, values):
        with open(path, mode="w") as f:
            f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            f_writer.writerow(heading)
            for value in values:
                f_writer.writerow(value)

    def finish(self):
        self.write_csv(Path(self.path, "train_loss.csv"), ["step", "train_loss"], [[key, value] for key, value in self.train_cost.items()])
        self.write_csv(Path(self.path, "val_loss.csv"), ["step", "val_loss"], [[key, value] for key, value in self.val_cost.items()])
        train_steps = list(self.train_cost.keys())
        simple_plot(train_steps, [self.train_cost[x] for x in train_steps], "Steps", "Train loss", Path(self.path, "train_loss.png"))
        val_steps = list(self.val_cost.keys())
        simple_plot(val_steps, [self.val_cost[x] for x in val_steps], "Steps", "Validation loss", Path(self.path, "val_loss.png"))

        for alg, result in self.testing.items():
            base_path = Path(self.path, alg)
            values = [[key] + [x for x in values] for key, values in result.items()]
            self.write_csv(Path(base_path, "clustering.csv"), CLUSTER_HEADING, values)
            self.generate_cluster_imgs(Path(self.path, alg), result)


    def generate_cluster_imgs(self, path, result):
        clustering_steps = list(result.keys())
        #simple_plot(clustering_steps, [result[x].accuracy for x in clustering_steps], "Steps", "Accuracy", Path(path, "accuracy.png"))
        simple_plot(clustering_steps, [result[x].rand_score for x in clustering_steps], "Epoch", "Rand score", Path(path, "rand_score.png"), [0,1.1])
        simple_plot(clustering_steps, [result[x].adjusted_rand_score for x in clustering_steps], "Epoch", "Adjusted rand score", Path(path, "adjusted_rand_score.png"), [-1.1, 1.1])
        simple_plot(clustering_steps, [result[x].calinski_harabasz for x in clustering_steps], "Epoch", "Calinski harabasz", Path(path, "calinski_harabasz.png"))
        simple_plot(clustering_steps, [result[x].davies_bouldin for x in clustering_steps], "Epoch", "Davies bouldin", Path(path, "davies_bouldin.png"))

    def visualize_cluster(self, X, labels, path):
        xs, ys = np.split(X, 2, axis=1)
        xs = np.reshape(xs, (xs.shape[0]))
        ys = np.reshape(ys, (ys.shape[0]))

        fig, ax = plt.subplots()
        a = ax.scatter(xs, ys, c=labels)
        fig.savefig(path)
        return