import json
from pennylane import numpy as np
from qclustering.JsonHelper import run_json_config

np.random.seed(42)

with open("config.json") as f:
    data = json.load(f)

qvk = run_json_config(data)