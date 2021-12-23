from typing import Tuple, Union

from qclustering.datasets.generated import circles
from qclustering.datasets.iris import iris
from qclustering.datasets.mnist import mnist
from qclustering.datasets.utils import DataSet


def load_data(
    dataset: str,
    train_size: int = 100,
    test_size: int = 50,
    val_size: int = 0,
    shuffle: Union[bool, int] = 1337,
    num_classes: int = 2,
    classes: Tuple[int, ...] = (6, 9),
    features: int = 4,
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
        return iris(train_size=train_size, test_size=test_size, val_size=val_size, shuffle=shuffle, num_classes=num_classes)
    elif dataset == "mnist":
        return mnist(
            wires=features,
            classes=classes,
            train_size=train_size,
            test_size=test_size,
            val_size=val_size,
            shuffle=shuffle,
        )
    elif dataset == "circles":
        return circles(train_size=train_size, test_size=test_size, val_size=val_size, shuffle=shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")