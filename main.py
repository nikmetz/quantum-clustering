from pennylane import numpy as np
from qclustering.JsonHelper import run_json_file

np.random.seed(42)

qvk = run_json_file("config.json")