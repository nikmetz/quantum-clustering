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

def donut(num_train):
    x_data, y_data = [], []

    inv_sqrt2 = 1/np.sqrt(2)
    x_donut = 1
    i = 0
    while (i<num_train):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            i += 1
            x_data.append([x_donut+x[0],x[1]])
            if r_squared < .25:
                y_data.append(x_donut)
            else:
                y_data.append(-x_donut)
            # Move over to second donut
            if i==num_train//2:
                x_donut = -1

    return np.array(x_data), np.array(y_data)

def symmetric_donuts(train_size=20, test_size=20, val_size=20):
    x_train, y_train = donut(train_size)
    x_test, y_test = donut(test_size)
    x_val, y_val = donut(val_size)

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)

def single_donut(num_train):
    x_data, y_data = [], []

    inv_sqrt2 = 1/np.sqrt(2)
    x_donut = 1
    i = 0
    while (i<num_train):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            if r_squared < 0.05:
                i += 1
                x_data.append(x)
                y_data.append(x_donut)
            elif r_squared > 0.45:
                i += 1
                x_data.append(x)
                y_data.append(-x_donut)

    return np.array(x_data), np.array(y_data)

def test_donuts(train_size=20, test_size=20, val_size=20):
    x_train, y_train = single_donut(train_size)
    x_test, y_test = single_donut(test_size)
    x_val, y_val = single_donut(val_size)

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)