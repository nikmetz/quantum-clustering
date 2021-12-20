import autograd.numpy as np
import matplotlib.pyplot as plt

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

def simple_plot(xs, ys, xlabel, ylabel, filename, ylimit=None):
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylimit is not None:
        ax.set_ylim(ylimit)
    plt.savefig(filename)
    return