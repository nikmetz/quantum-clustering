from qclustering.datasets.generated import generate
from qclustering.datasets.iris import iris
from qclustering.datasets.mnist import mnist
from qclustering.datasets.utils import DataSet


def load_data(
    dataset: str,
    **dataset_params
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
        return iris(**dataset_params)
    elif dataset == "mnist":
        return mnist(**dataset_params)
    elif dataset in ["circles", "moons", "classification", "blobs"]:
        return generate(dataset=dataset, **dataset_params)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")