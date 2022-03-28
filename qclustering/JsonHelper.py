import json
import os
import shutil
import qclustering.datasets
import time
import datetime
import pickle
from pathlib import Path
from qclustering.QuantumVariationalKernel import OverlapQuantumVariationalKernel, SwapQuantumVariationalKernel
from qclustering.utils import get_params
from pennylane import numpy as np
from qclustering.Ansatz import get_ansatz
from qclustering.Logging import Logging
from qclustering.datasets.utils import DataSet
from qclustering.KernelMatrix import set_postprocessing_parameters, set_processing_pool

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
    num_processes = js.get("num_processes", 1)

    np.random.seed(numpy_seed)

    init_params = get_params(strategy=js.get("init_params"), num_wires=wires, num_layers=layers, params_per_wire=params_per_wire)

    ansatz = get_ansatz(ansatz_name, layers, ansatz_params)

    dataset_params = js.get("dataset_params", {})

    set_postprocessing_parameters(js.get("postprocessing_functions", []), js.get("postprocessing_parameters", []))

    if num_processes > 1:
        set_processing_pool(num_processes)

    data = qclustering.datasets.load_data(
        dataset = js.get("dataset"),
        two_classes = dataset_params.pop("two_classes", False),
        scale_factors = dataset_params.pop("scale_factors", []),
        scale_pi = dataset_params.pop("scale_pi", False),
        dataset_params = dataset_params
    )

    if "train_data" in js and "train_target" in js:
        train_data = np.array(js.get("train_data"))
        train_target = np.array(js.get("train_target"))
        data = DataSet(train_data, train_target, data.validation_data, data.validation_target, data.test_data, data.test_target)

    logging_obj = Logging(
        data = data,
        test_clustering_interval = js.get("test_clustering_interval", 100),
        train_clustering_interval= js.get("train_clustering_interval", 100),
        testing_algorithms = js.get("testing_algorithms", ["kmeans"]),
        testing_algorithm_params = js.get("testing_algorithm_params", [{}]),
        path = path
    )
    if circuit == "overlap":
        qvk = OverlapQuantumVariationalKernel(wires, ansatz, init_params, device, shots, logging_obj)
    elif circuit == "swap":
        noise_name = js.get("noise_model", None)
        if noise_name is not None:
            noise_models_list = pickle.load(open(Path("qclustering/ibmq_noise_models.p"), "rb"))
            if not noise_name in noise_models_list:
                raise ValueError(f"Unknown noise model: {noise_name}")
            noise_model = noise_models_list[noise_name]
        else:
            noise_model = None
        qvk = SwapQuantumVariationalKernel(wires, ansatz, init_params, device, shots, noise_model, logging_obj)
    else:
        raise ValueError(f"Unknown circuit name: {circuit}")

    qvk.train(
        data = data,
        batch_size = js.get("batch_size", 5),
        num_batches = js.get("num_batches", None),
        epochs = js.get("epochs", 500),
        learning_rate = js.get("learning_rate", 0.2),
        learning_rate_decay = js.get("learning_rate_decay", 0.0),
        optimizer_name = js.get("optimizer", "GradientDescent"),
        optimizer_params = js.get("optimizer_params", {}),
        cost_func_name = js.get("cost_func", "KTA-supervised"),
        cost_func_params = js.get("cost_func_params", {})
    )

    return qvk