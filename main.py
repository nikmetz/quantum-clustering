import numpy as np
from qclustering.datasets.iris import iris
from qclustering.QuantumVariationalKernel import QuantumVariationalKernel
from qclustering.QuantumVariationalKernel import random_params
from qclustering.Ansatz import ansatz2

def euclidean_dist(x, y):
    return np.linalg.norm(x - y)

np.random.seed(42)
data = iris(train_size=55, test_size=0, shuffle=True)

wires = 4
layers = 3

init_params = random_params(num_wires=wires, num_layers=layers, params_per_wire=2)
qvk = QuantumVariationalKernel(wires=wires, ansatz=ansatz2, init_params=init_params, cost_func="triplet-loss-supervised")

qvk.train(data.train_data, data.train_target)