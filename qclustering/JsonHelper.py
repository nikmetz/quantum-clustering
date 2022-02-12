import json
import os
import shutil
import qclustering.datasets
import time
import datetime
import multiprocessing
from qclustering.QuantumVariationalKernel import OverlapQuantumVariationalKernel, SwapQuantumVariationalKernel
from qclustering.utils import get_params
from pennylane import numpy as np
from qclustering.Ansatz import get_ansatz
from qclustering.Logging import Logging

def run_json_file(file):
    with open(file) as f:
        js = json.load(f)

    st = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = os.path.basename(file).split(".")[0] + "_" + st + "/"
    os.makedirs(folder_name, exist_ok=True)
    shutil.copy(file, folder_name)

    run_json_config(js, folder_name)

def run_json_config(js, path=""):
    circuit = js.get("circuit", "overlap")
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

    num_processes = js.get("num_processes", 1)

    if num_processes > 1:
        pool = multiprocessing.Pool(processes=num_processes)
    else:
        pool = None

    dataset_params = js.get("dataset_params", {})

    data = qclustering.datasets.load_data(
        dataset = js.get("dataset"),
        two_classes = dataset_params.pop("two_classes", False),
        scale = dataset_params.pop("scale", 0),
        dataset_params = dataset_params
    )

    logging_obj = Logging(
        testing_interval = js.get("clustering_interval", 100),
        testing_algorithms = js.get("clustering_algorithm", ["kmeans"]),
        testing_algorithm_params = js.get("clustering_algorithm_params", [{}]),
        process_pool = pool,
        path = path
    )
    if circuit == "overlap":
        qvk = OverlapQuantumVariationalKernel(wires, ansatz, init_params, device, shots, logging_obj)
    elif circuit == "swap":
        qvk = SwapQuantumVariationalKernel(wires, ansatz, init_params, device, shots, logging_obj)
    else:
        raise ValueError(f"Unknown circuit name: {circuit}")

    qvk.train(
        data = data,
        batch_size = js.get("batch_size", 5),
        epochs = js.get("epochs", 500),
        learning_rate = js.get("learning_rate", 0.2),
        learning_rate_decay = js.get("learning_rate_decay", 0.0),
        optimizer_name = js.get("optimizer", "GradientDescent"),
        optimizer_params = js.get("optimizer_params", {}),
        cost_func_name = js.get("cost_func", "KTA-supervised"),
        cost_func_params = js.get("cost_func_params", {})
    )

    return qvk