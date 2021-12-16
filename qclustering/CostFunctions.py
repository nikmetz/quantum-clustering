from pennylane import numpy as np
from qclustering.utils import vector_change
import pennylane as qml

def multiclass_target_alignment(
    X,
    Y,
    kernel,
    num_classes,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    num_samples = Y.shape[0]
    if rescale_class_labels:
        unique, counts = np.unique(Y, return_counts=True)
        test = [counts[np.argwhere(unique==x)[0][0]] for x in Y]
        scale = np.outer(test, test)
    else:
        scale = np.ones((num_samples, num_samples))

    T = None
    for i in range(num_samples):
        n = [[1/scale[i][j] if Y[i]==Y[j] else (-1/(num_classes-1))/scale[i][j] for j in range(num_samples)]]
        if T is None:
            T = np.copy(n)
        else:
            T = np.concatenate((T, n), axis=0)

    return qml.math.frobenius_inner_product(K, T, normalize=True)

def triplet_loss(X, labels, alpha, dist_func):
    sum = 0.0
    K = qml.kernels.square_kernel_matrix(X, dist_func, assume_normalized_kernel=False)
    for anchor_idx, anchor in enumerate(X):
        current = K[anchor_idx]
        #distance between anchor and the sample with a different label than the anchor and biggest distance to the anchor
        dist_anchor_negative = current[labels != labels[anchor_idx]].max()

        positive_mask = labels == labels[anchor_idx]
        if len(current[positive_mask]) > 1:
            #if there is more than one sample with one label, don't choose the anchor as the positive example
            positive_mask = vector_change(positive_mask, False, anchor_idx)
        #distance between anchor and the sample with the same label as the anchor and smallest distance to the anchor
        dist_anchor_positive = current[positive_mask].min()
        
        sum = sum + max(dist_anchor_positive - dist_anchor_negative + alpha, 0)
    return sum