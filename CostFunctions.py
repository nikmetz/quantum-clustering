from pennylane import numpy as np
import pennylane as qml
import random

def multiclass_target_alignment(
    X,
    Y,
    kernel,
    num_classes,
    assume_normalized_kernel=False,
):

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    num_samples = Y.shape[0]
    T = [[1 if Y[0] == Y[j] else (-1/(num_classes-1)) for j in range(num_samples)]]
    if num_samples.shape[0] > 1:
        for i in range(1, num_samples):
            T = np.concatenate((T, [[1. if Y[i] == Y[j] else (-1/(num_classes-1)) for j in range(num_samples)]]), axis=0)

    return qml.math.frobenius_inner_product(K, T, normalize=True)

def triplet_loss(X, labels, alpha, dist_func):
    sum = 0.0
    for anchor_idx, anchor in enumerate(X):
        negative = random.choice(X[labels != labels[anchor_idx]])
        positive_mask = labels == labels[anchor_idx]
        #positive_mask = vector_change(positive_mask, False, anchor_index)
        positive = random.choice(X[positive_mask])
        sum = sum + max(dist_func(anchor, positive) - dist_func(anchor, negative) + alpha, 0)
    return sum