from pennylane import numpy as np
import pennylane as qml
import random

def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product

def triplet_loss(X, labels, num_samples, alpha, dist_func):
    sum = 0.0
    for i in range(num_samples):
        anchor_index = random.randrange(labels.shape[0])
        anchor = X[anchor_index]
        negative = random.choice(X[labels != labels[anchor_index]])
        positive_mask = labels == labels[anchor_index]
        #positive_mask = vector_change(positive_mask, False, anchor_index)
        positive = random.choice(X[positive_mask])
        sum = sum + max(dist_func(anchor, positive) - dist_func(anchor, negative) + alpha, 0)
    return sum