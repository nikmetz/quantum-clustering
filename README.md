# Quantum-Clustering

Quantum-clustering is a framework to train and test variational quantum kernel (VQK) functions in cooperation with classical clustering algorithms using pennylane for the quantum simulation and scikit-learn for the clustering algorithms. In order to test this concept, this framework supports many different different parameters, including cost functions, optimizers, quantum ansätze, datasets, clustering algorithms, etc.

## Basic Usage

There are two main ways to use this framework, directly from the code or with configuration files.

### Code

Creating the necessary python objects and starting the training by hand. Preferred if you only want to use parts of the existing framework for other purposes.

### Configuration Files

Define a Json configuration file using the [configuration options](##configuration-options) and execute the config file with the provided functions. Preferred for large scale test scenarios.

```python
from qclustering.JsonHelper import run_json_file
run_json_file("config.json")
```

## Configuration Options

The ```config.json``` file shows an example configuration file. The following list shows most of the possible configuration details.

- ```numpy_seed: int``` - Numpy seed to achieve reproducable results for the randomness
- ```shots: int``` - Number of shots used for the quantum calculations
- ```device: string = ["default.qubit"|"default.mixed"]``` - Quantum device used for the simulation.
- ```circuit: string = ["overlap"|"swap"]``` - Quantum circuit used for calculating the kernel function. Either "overlap" or "swap"
- ```num_processes: int``` - Number of processes used for calculating the full kernel matrix. Currently only supported for noiseless calculations.
- ```cost_func: string = ["KTA"|"triplet-loss"|"davies-bouldin"|"calinski-harabasz"]``` - Name of the cost function used to train the VQK
- ```cost_func_params: Object``` - All further parameters for the given cost function.
- ```ansatz: string = ["ansatz1"|"ansatz1noise"|"ansatz2"]``` - Name of the quantum ansatz to use.
- ```ansatz_params: Object``` - Further configuration for the ansatz.
  - ```wires: int``` - Number of qubits to use.
  - ```layers: int``` - Number of layers to use.
- ```dataset: string = ["iris"|"moons"|"donuts"]``` - Name of the dataset to use.
- ```dataset_params: Object``` - Further parameters for the dataset.
  - ```train_size: int``` - Number of samples to use for training.
  - ```val_size: int``` - Number of samples to use for validation.
  - ```test_size: int``` - Number of samples to use for testing.
  - ```apply_pca: bool``` - Apply PCA to the dataset.
  - ```num_features: int``` - Define the desired number of features for PCA.
- ```epochs: int``` - Number of epochs to train the VQK.
- ```batch_size: int``` - Number of samples in each batch.
- ```learning_rate: float``` - Step size for the optimizer.
- ```learning_rate_decay: float``` - Change of the learning rate in each epoch.
- ```optimizer: string = ["GradientDescent"|"Adam"]``` - Name of the optimizer to use.
- ```optimizer_params: Object``` - Further paramters for the optimizer to use.
- ```test_clustering_interval: int``` - Interval at which the VQK is used for clustering the testing data during the training process.
- ```testing_algorithms: Array = ["kmeans"|"spectral"|"dbscan"|"agglomerative"]``` - List of the names of clustering algorithms to use for testing.
- ```testing_algorithm_params: Array``` - List of objects. Each object is used as further parameters for each algorithm defined in ```testing_algorithms```.
