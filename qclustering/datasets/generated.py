from sklearn import datasets
from qclustering.datasets.utils import DataSet

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

    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[train_size+val_size:train_size+val_size+test_size], y[train_size+val_size:train_size+val_size+test_size]

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)