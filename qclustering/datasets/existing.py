from pennylane import numpy as np
from sklearn import datasets
from qclustering.datasets.utils import DataSet

def load_existing(dataset, train_size=50, val_size=5, test_size=50, shuffle=True, classes=[0,1,2]) -> DataSet:
    if dataset == "iris":
        data, target = datasets.load_iris(return_X_y=True)
    elif dataset == "wine":
        data, target = datasets.load_wine(return_X_y=True)
    elif dataset == "breast-cancer":
        data, target = datasets.load_breast_cancer(return_X_y=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if dataset in ["iris", "wine"]:
        if len(classes) < 3:
            mask = [True if y in classes else False for y in target]
            data, target = data[mask], target[mask]

    if shuffle:
        perm = np.random.permutation(data.shape[0])
        data, target = data[perm], target[perm]

    num_samples = len(data)

    x_train, y_train = data[:train_size], target[:train_size]
    x_val, y_val = data[train_size:train_size+val_size], target[train_size:train_size+val_size]
    if train_size + test_size + val_size > num_samples:
        x_test, y_test = data[:test_size], target[:test_size]
    else:
        x_test, y_test = data[train_size+val_size:train_size+val_size+test_size], target[train_size+val_size:train_size+val_size+test_size]

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)