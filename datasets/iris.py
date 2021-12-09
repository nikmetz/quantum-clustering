from pennylane import numpy as np
from sklearn import datasets
from datasets.utils import one_hot, DataSet

MAX_SAMPLES = 150


def iris(train_size=100, test_size=50, shuffle=True) -> DataSet:
    train_size = min(train_size, MAX_SAMPLES)
    if train_size + test_size > MAX_SAMPLES:
        test_size = MAX_SAMPLES - train_size
    data, target = datasets.load_iris(return_X_y=True)
    mask = [True if y == 0 or y == 1 else False for y in target]
    data, target = data[mask], target[mask]

    if shuffle:
        perm = np.random.permutation(data.shape[0])
        data, target = data[perm], target[perm]

    x_train, y_train = data[:train_size], target[:train_size]
    x_test, y_test = data[train_size:], target[test_size:]

    return DataSet(x_train, y_train, None, None, x_test, y_test)