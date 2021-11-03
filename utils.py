import numpy as np
import matplotlib.pyplot as plt

def visualize(X, true_labels, labels):
    mark = [".", "^", "s", "P", ""]
    xs, ys = np.split(X, 2, axis=1)
    xs = np.reshape(xs, (xs.shape[0]))
    ys = np.reshape(ys, (ys.shape[0]))

    fig, ax = plt.subplots()
    num_clusters = np.max(true_labels) + 1
    for i in range(num_clusters):
        a = ax.scatter(xs[true_labels == i], ys[true_labels == i], marker=mark[i], c=labels[true_labels == i])
    plt.show()