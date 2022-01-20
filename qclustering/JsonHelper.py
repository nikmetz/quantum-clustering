import json
import os
import shutil
import qclustering.datasets
import time
import datetime
from qclustering.QuantumVariationalKernel import QuantumVariationalKernel
from qclustering.utils import get_params
from pennylane import numpy as np
from qclustering.Ansatz import get_ansatz

def run_json_file(file):
    with open(file) as f:
        js = json.load(f)

    st = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.basename(file).split(".")[0] + "_" + st + "/"
    os.makedirs(folder_name, exist_ok=True)
    shutil.copy(file, folder_name)

    run_json_config(js, folder_name)

def run_json_config(js, path=""):
    ansatz_name = js.get("ansatz", "ansatz")
    ansatz_params = js.get("ansatz_params", {})
    wires = ansatz_params.pop("wires", 4)
    layers = ansatz_params.pop("layers", 3)
    params_per_wire = ansatz_params.pop("params_per_wire", 2)
    cost_func = js.get("cost_func")
    device = js.get("device", "default.qubit")
    shots = js.get("shots", None)
    numpy_seed = js.get("numpy_seed", 1546)

    np.random.seed(numpy_seed)

    init_params = get_params(strategy=js.get("init_params"), num_wires=wires, num_layers=layers, params_per_wire=params_per_wire)

    ansatz = get_ansatz(ansatz_name, wires, layers, ansatz_params)

    qvk = QuantumVariationalKernel(wires, ansatz, init_params, cost_func, device, shots)

    data = qclustering.datasets.load_data(js.get("dataset"), **js.get("dataset_params", {}))

    qvk.train(data, path=path, **js)

    return qvk