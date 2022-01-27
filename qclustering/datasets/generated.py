from sklearn import datasets
from qclustering.datasets.utils import DataSet
from sklearn.preprocessing import minmax_scale
from pennylane import numpy as np

def generate(dataset, **dataset_params) -> DataSet:
    train_size = dataset_params.pop("train_size", 50)
    val_size = dataset_params.pop("val_size", 50)
    test_size = dataset_params.pop("test_size", 50)

    if dataset == "circles":
        x, y = datasets.make_circles(n_samples=train_size + test_size + val_size, **dataset_params)
    elif dataset == "moons":
        x, y = datasets.make_moons(n_samples=train_size + test_size + val_size, **dataset_params)
    elif dataset == "classification":
        x, y = datasets.make_classification(n_samples=train_size + test_size + val_size, **dataset_params)
    elif dataset == "blobs":
        x, y = datasets.make_blobs(n_samples=train_size + test_size + val_size, **dataset_params)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    x_train, y_train = minmax_scale(x[:train_size], (0, 2 * np.pi)), y[:train_size]
    x_val, y_val = minmax_scale(x[train_size:train_size+val_size], (0, 2 * np.pi)), y[train_size:train_size+val_size]
    x_test, y_test = minmax_scale(x[train_size+val_size:train_size+val_size+test_size], (0, 2 * np.pi)), y[train_size+val_size:train_size+val_size+test_size]

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)