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
    ansatz_name = js.get("ansatz", "ansatz1")
    ansatz_params = js.get("ansatz_params", {})
    wires = ansatz_params.pop("wires", 4)
    layers = ansatz_params.pop("layers", 3)
    params_per_wire = ansatz_params.pop("params_per_wire", 2)
    device = js.get("device", "default.qubit")
    shots = js.get("shots", None)
    numpy_seed = js.get("numpy_seed", 1546)

    np.random.seed(numpy_seed)

    init_params = get_params(strategy=js.get("init_params"), num_wires=wires, num_layers=layers, params_per_wire=params_per_wire)

    ansatz = get_ansatz(ansatz_name, wires, layers, ansatz_params)

    qvk = QuantumVariationalKernel(wires, ansatz, init_params, device, shots)

    data = qclustering.datasets.load_data(js.get("dataset"), **js.get("dataset_params", {}))

    qvk.train(
        data = data,
        batch_size = js.get("batch_size", 5),
        epochs = js.get("epochs", 500),
        learning_rate = js.get("learning_rate", 0.2),
        learning_rate_decay = js.get("learning_rate_decay", 0.0),
        clustering_interval = js.get("clustering_interval", 100),
        optimizer_name = js.get("optimizer", "GradientDescent"),
        optimizer_params = js.get("optimizer_params", {}),
        clustering_algorithm = js.get("clustering_algorithm", "kmeans"),
        clustering_algorithm_params = js.get("clustering_algorithm_params", {}),
        cost_func_name = js.get("cost_func", "KTA-supervised"),
        cost_func_params = js.get("cost_func_params", {}),
        logging_path = path
    )

    return qvk