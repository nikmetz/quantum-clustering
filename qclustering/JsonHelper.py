import qclustering.Ansatz as Ansatz
from qclustering.datasets.iris import iris
from qclustering.QuantumVariationalKernel import QuantumVariationalKernel
from qclustering.utils import random_params
from qclustering.utils import fixed_value_params
from pennylane import numpy as np

def run_json_config(js):
    wires = js.get("wires")
    layers = js.get("layers")
    params_per_wire = js.get("params_per_wire")
    cost_func = js.get("cost_func")
    device = js.get("device", "default.qubit")
    shots = js.get("shots", None)

    if js.get("init_params") == "random":
        init_params = random_params(wires, layers, params_per_wire)
    elif js.get("init_params") == "fixed":
        init_params = fixed_value_params(np.pi, wires, layers, params_per_wire)

    if js.get("ansatz") == "ansatz":
        ansatz = Ansatz.ansatz
    elif js.get("ansatz") == "ansatz2":
        ansatz = Ansatz.ansatz2

    qvk = QuantumVariationalKernel(wires, ansatz, init_params, cost_func, device, shots)

    if js.get("dataset") == "iris":
        data = iris(train_size=25, test_size=0, shuffle=True)

    qvk.train(data.train_data, data.train_target, **js)

    return qvk