import qclustering.Ansatz as Ansatz
import json
import os
import shutil
from qclustering.datasets.iris import iris
from qclustering.QuantumVariationalKernel import QuantumVariationalKernel
from qclustering.utils import random_params
from qclustering.utils import fixed_value_params
from pennylane import numpy as np

def run_json_file(file):
    with open(file) as f:
        js = json.load(f)

    folder_name = os.path.basename(file).split(".")[0]+"/"
    os.makedirs(folder_name, exist_ok=True)
    shutil.copy(file, folder_name)

    run_json_config(js, folder_name)

def run_json_config(js, path=""):
    wires = js.get("wires")
    layers = js.get("layers")
    params_per_wire = js.get("params_per_wire")
    cost_func = js.get("cost_func")
    device = js.get("device", "default.qubit")
    shots = js.get("shots", None)
    train_size = js.get("train_size", 20)
    val_size = js.get("val_size", 5)
    test_size = js.get("test_size", 25)

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
        data = iris(train_size=train_size, val_size=val_size, test_size=test_size, shuffle=True)

    qvk.train(data, path=path, **js)

    return qvk