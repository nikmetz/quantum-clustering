from pennylane import numpy as np
from sklearn import datasets
from qclustering.datasets.utils import DataSet

MAX_SAMPLES = 150


def iris(train_size=95, val_size=5, test_size=50, shuffle=True, num_classes=3) -> DataSet:
    if train_size + test_size + val_size > MAX_SAMPLES:
        raise ValueError("Requested too many samples from the iris dataset. Maximum is: {}, but {} were requested.".format(MAX_SAMPLES, train_size + test_size + val_size))
    data, target = datasets.load_iris(return_X_y=True)
    if num_classes == 2:
        if train_size + test_size + val_size > 100:
            raise ValueError("Requested too many samples from the iris dataset with two classes. Maximum is: {}, but {} were requested.".format(100, train_size + test_size + val_size))
        mask = [True if y == 0 or y == 1 else False for y in target]
        data, target = data[mask], target[mask]

    if shuffle:
        perm = np.random.permutation(data.shape[0])
        data, target = data[perm], target[perm]

    x_train, y_train = data[:train_size], target[:train_size]
    x_val, y_val = data[train_size:train_size+val_size], target[train_size:train_size+val_size]
    x_test, y_test = data[train_size+val_size:train_size+val_size+test_size], target[train_size+val_size:train_size+val_size+test_size]

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)