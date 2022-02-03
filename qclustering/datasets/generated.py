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

def single_donut(num_samples, num_donuts, donut_inner_distance = 0.25):
    if donut_inner_distance < 0.0 or donut_inner_distance >= 0.5:
        raise ValueError(f"Donut inner distance has to be in range [0.0, 0.5). {donut_inner_distance} is not allowed.")

    x_data, y_data = [], []

    inv_sqrt2 = 1/np.sqrt(2)
    x_donut = 1
    i = 0
    while (i<num_samples):
        x = np.random.uniform(-inv_sqrt2,inv_sqrt2, 2)
        r_squared = np.linalg.norm(x, 2)**2
        if r_squared < 0.5:
            if r_squared < 0.0 + (0.5 - donut_inner_distance)/2:
                i += 1
                x_data.append([x_donut+x[0],x[1]])
                y_data.append(x_donut)
            elif r_squared > 0.5 - (0.5 - donut_inner_distance)/2:
                i += 1
                x_data.append([x_donut+x[0],x[1]])
                y_data.append(-x_donut)
            if x_donut == 1 and num_donuts == 2 and i==num_samples//2:
                x_donut = -1

    return np.array(x_data), np.array(y_data)

def donuts(
    train_size = 20,
    test_size = 20,
    val_size = 20,
    num_donuts = 2,
    train_donut_inner_distance = 0.25,
    val_donut_inner_distance = None,
    test_donut_inner_distance = None,
):
    if val_donut_inner_distance is None:
        val_donut_inner_distance = train_donut_inner_distance
    if test_donut_inner_distance is None:
        test_donut_inner_distance = train_donut_inner_distance
    
    x_train, y_train = single_donut(train_size, num_donuts, train_donut_inner_distance)
    x_test, y_test = single_donut(test_size, num_donuts, test_donut_inner_distance)
    x_val, y_val = single_donut(val_size, num_donuts, val_donut_inner_distance)

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)