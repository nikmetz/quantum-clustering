from qclustering.datasets.generated import generate
from qclustering.datasets.generated import donuts
from qclustering.datasets.iris import iris
from qclustering.datasets.mnist import mnist
from qclustering.datasets.utils import DataSet
from pennylane import numpy as np


def load_data(
    dataset: str,
    two_classes = False,
    dataset_params = {}
) -> DataSet:
    """
    Returns the data for the requested ``dataset``.
    :param dataset: Name of the chosen dataset.
        Available datasets are: iris, mnist and circles.
    :param train_size: Size of the training dataset
    :param test_size: Size of the testing dataset
    :param val_size: Size of the validation dataset
    :param shuffle: if the dataset should be shuffled, used also as a seed
    :param num_classes: number of different classes that should be included
    :param classes: which numbers of the mnist dataset should be included
    :param features: number of features in the data
    :raises ValueError: Raised if a not supported dataset is requested
    """

    if dataset == "iris":
        data = iris(**dataset_params)
    elif dataset == "mnist":
        data = mnist(**dataset_params)
    elif dataset in ["circles", "moons", "classification", "blobs"]:
        data = generate(dataset=dataset, **dataset_params)
    elif dataset == "donuts":
        data = donuts(**dataset_params)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if two_classes:
        train = np.array([1 if x == data.train_target[0] else -1 for x in data.train_target])
        test = np.array([1 if x == data.train_target[0] else -1 for x in data.test_target])
        val = np.array([1 if x == data.train_target[0] else -1 for x in data.validation_target])
        data = DataSet(data.train_data, train, data.validation_data, val, data.test_data, test)

    return data