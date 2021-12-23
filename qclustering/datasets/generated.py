from sklearn import datasets
from qclustering.datasets.utils import DataSet

def circles(train_size=100, test_size=50, val_size=50, shuffle=True) -> DataSet:
    x, y = datasets.make_circles(n_samples=train_size + test_size + val_size, shuffle=shuffle)
    x_train, y_train = x[:train_size], y[:train_size]
    x_val, y_val = x[train_size:train_size+val_size], y[train_size:train_size+val_size]
    x_test, y_test = x[train_size+val_size:train_size+val_size+test_size], y[train_size+val_size:train_size+val_size+test_size]

    return DataSet(x_train, y_train, x_val, y_val, x_test, y_test)