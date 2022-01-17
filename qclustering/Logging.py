from typing import NamedTuple, Dict
from dataclasses import dataclass, field
from sklearn import metrics
from qclustering.utils import simple_plot
from pennylane import numpy as np
import csv

class ClusterResult(NamedTuple):
    rand_score: float
    adjusted_rand_score: float
    mutual_info: float
    adjusted_mutual_info: float
    normalized_mutual_info: float
    fowlkes_mallows_score: float
    calinski_harabasz: float
    davies_bouldin: float

def result_to_list(result: ClusterResult):
    return [result.rand_score, result.adjusted_rand_score, result.mutual_info, result.adjusted_mutual_info, result.normalized_mutual_info, result.fowlkes_mallows_score, result.calinski_harabasz, result.davies_bouldin]

def get_cluster_result(X, labels_true, labels_pred) -> ClusterResult:
    rand_score = metrics.rand_score(labels_true, labels_pred)
    adjusted_rand_score = metrics.adjusted_rand_score(labels_true, labels_pred)
    mutual_info = metrics.mutual_info_score(labels_true, labels_pred)
    adjusted_mutual_info = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    normalized_mutual_info = metrics.normalized_mutual_info_score(labels_true, labels_pred)
    fowlkes_mallows_score = metrics.fowlkes_mallows_score(labels_true, labels_pred)
    if len(np.unique(labels_pred)) < 2:
        calinski_harabasz = 0
        davies_bouldin = 0
    else:
        calinski_harabasz = metrics.calinski_harabasz_score(X, labels_pred)
        davies_bouldin = metrics.davies_bouldin_score(X, labels_pred)

    return ClusterResult(rand_score, adjusted_rand_score, mutual_info, adjusted_mutual_info, normalized_mutual_info, fowlkes_mallows_score, calinski_harabasz, davies_bouldin)

@dataclass
class Logging:
    train_cost: Dict = field(default_factory=dict)
    val_cost: Dict = field(default_factory=dict)
    clustering: Dict = field(default_factory=dict)

    def log_training(self, step: int, cost: float, val_cost: float):
        self.train_cost[step] = cost
        self.val_cost[step] = val_cost

    def log_clustering(self, step: int, result: ClusterResult):
        self.clustering[step] = result

    def write_csv(self, path=""):
        with open(path+"train_loss.csv", mode="w") as train_loss:
            train_loss_writer = csv.writer(train_loss, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            train_loss_writer.writerow(["step", "train_loss"])
            for key, value in self.train_cost.items():
                train_loss_writer.writerow([key, value])

        with open(path+"val_loss.csv", mode="w") as val_loss:
            val_loss_writer = csv.writer(val_loss, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            val_loss_writer.writerow(["step", "val_loss"])
            for key, value in self.val_cost.items():
                val_loss_writer.writerow([key, value])

        with open(path+"clustering.csv", mode="w") as clustering_f:
            clustering_writer = csv.writer(clustering_f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            clustering_writer.writerow(["step", "rand_score", "adjusted_rand_score", "mutual_info", "adjusted_mutual_info", "normalized_mutual_info", "fowlkes_mallows_score", "calinksi_harabasz", "davies_bouldin"])
            for key, value in self.clustering.items():
                clustering_writer.writerow([key]+result_to_list(value))

    def generate_imgs(self, path=""):
        train_steps = list(self.train_cost.keys())
        simple_plot(train_steps, [self.train_cost[x] for x in train_steps], "Steps", "Train loss", path+"train_loss.png")
        val_steps = list(self.val_cost.keys())
        simple_plot(val_steps, [self.val_cost[x] for x in val_steps], "Steps", "Validation loss", path+"val_loss.png")

        clustering_steps = list(self.clustering.keys())
        simple_plot(clustering_steps, [self.clustering[x].rand_score for x in clustering_steps], "Steps", "Rand score", path+"rand_score.png")
        simple_plot(clustering_steps, [self.clustering[x].adjusted_rand_score for x in clustering_steps], "Steps", "Adjusted rand score", path+"adjusted_rand_score.png")
        simple_plot(clustering_steps, [self.clustering[x].mutual_info for x in clustering_steps], "Steps", "Mutual info", path+"mutual_info.png")
        simple_plot(clustering_steps, [self.clustering[x].adjusted_mutual_info for x in clustering_steps], "Steps", "Adjusted mutual info", path+"adjusted_mutual_info.png")
        simple_plot(clustering_steps, [self.clustering[x].normalized_mutual_info for x in clustering_steps], "Steps", "Normalized mutual info", path+"normalized_mutual_info.png")
        simple_plot(clustering_steps, [self.clustering[x].fowlkes_mallows_score for x in clustering_steps], "Steps", "Fowlkes mallows score", path+"fowlkes_mallows_score.png")
        simple_plot(clustering_steps, [self.clustering[x].calinski_harabasz for x in clustering_steps], "Steps", "Calinski harabasz", path+"calinski_harabasz.png")
        simple_plot(clustering_steps, [self.clustering[x].davies_bouldin for x in clustering_steps], "Steps", "Davies bouldin", path+"davies_bouldin.png")