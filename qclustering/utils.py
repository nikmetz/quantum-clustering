from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from pathlib import Path
import qiskit.providers.aer.noise as noise
import pickle
import pennylane as qml

def build_noise_model(noise_model_params, num_qubits):
    if noise_model_params is None:
        return None
    
    if type(noise_model_params) is str:
        noise_models_list = pickle.load(open(Path("qclustering/ibmq_noise_models.p"), "rb"))
        if not noise_model_params in noise_models_list:
            raise ValueError(f"Unknown noise model: {noise_model_params}")
        return noise_models_list[noise_model_params]

    elif type(noise_model_params) is list:
        noise_model = noise.NoiseModel()
        for noise_instance in noise_model_params:
            name = noise_instance["name"]
            parameter = noise_instance.get("parameter", None)
            gates = noise_instance.get("gates", None)

            if name == "coherent-x":
                error = noise.coherent_unitary_error(qml.matrix(qml.RX(parameter * 2*np.pi, wires=0)))
                gates = ['x'] if gates is None else gates
            elif name == "coherent-y":
                error = noise.coherent_unitary_error(qml.matrix(qml.RY(parameter * 2*np.pi, wires=0)))
                gates = ['y'] if gates is None else gates
            elif name == "coherent-z":
                error = noise.coherent_unitary_error(qml.matrix(qml.RZ(parameter * 2*np.pi, wires=0)))
                gates = ['z'] if gates is None else gates
            elif name == "bit-flip":
                error = noise.pauli_error([('X', parameter), ('I', 1-parameter)])
                gates = ['x', 'y', 'z', 'h'] if gates is None else gates
            elif name == "phase-damp":
                error = noise.phase_damping_error(parameter)
                gates = ['x', 'y', 'z', 'h'] if gates is None else gates
            elif name == "amplitude-damp":
                error = noise.amplitude_damping_error(parameter)
                gates = ['x', 'y', 'z', 'h'] if gates is None else gates
            elif name == "depolarizing":
                error = noise.depolarizing_error(parameter, 1)
                gates = ['x', 'y', 'z', 'h'] if gates is None else gates
            else:
                raise ValueError(f"Unknown noise name: {name}")

            noise_model.add_all_qubit_quantum_error(error, gates)

        return noise_model

    return None

def get_params(strategy="random", **params):
    if strategy == "random":
        return random_params(**params)
    elif strategy == "fixed":
        return fixed_value_params(**params)
    else:
        return ValueError(f"Unknown parameters initialization strategy: {strategy}")

def fixed_value_params(value, num_wires, num_layers, params_per_wire):
    return np.full((num_layers, num_wires, params_per_wire), value)

def random_params(num_wires, num_layers, params_per_wire):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, num_wires, params_per_wire))

def visualize(X, true_labels, labels):
    mark = [".", "^", "s", "P", ""]
    colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    xs, ys = np.split(X, 2, axis=1)
    xs = np.reshape(xs, (xs.shape[0]))
    ys = np.reshape(ys, (ys.shape[0]))

    fig, ax = plt.subplots()
    clusters = np.unique(true_labels)
    for idx, val in enumerate(clusters):
        a = ax.scatter(xs[true_labels == val], ys[true_labels == val], marker=mark[idx], c=colors[labels[true_labels == val]])
    plt.show()

def visualize_cluster(X, labels, path):
    xs, ys = np.split(X, 2, axis=1)
    xs = np.reshape(xs, (xs.shape[0]))
    ys = np.reshape(ys, (ys.shape[0]))

    fig, ax = plt.subplots()
    a = ax.scatter(xs, ys, c=labels)
    fig.savefig(path)
    return

def simple_plot(xs, ys, xlabel, ylabel, filename, ylimit=None):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    fig.savefig(filename)
    return

def plot_kernel(kernel_mat, y_test, epoch, path):
    embedding = MDS(n_components=2, dissimilarity="precomputed")
    dissimilarities = 1 - kernel_mat
    X_transformed = embedding.fit_transform(dissimilarities, y_test)

    fig, ax = plt.subplots()
    ax.scatter(X_transformed[:,0], X_transformed[:,1], c=y_test)
    ax.set_ylim([-1,1])
    ax.set_xlim([-1,1])
    fig.suptitle("Epoch: "+str(epoch))
    fig.savefig(path)
    return