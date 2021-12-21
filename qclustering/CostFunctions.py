from pennylane import numpy as np
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

def triplet_loss(X, labels, dist_func, **kwargs):
    strategy = kwargs.get("strategy", "random")
    alpha = kwargs.get("alpha")

    sum = 0.0
    if strategy == "minmax":
        #only needed for minmax strategy, not random
        K = qml.kernels.square_kernel_matrix(X, dist_func, assume_normalized_kernel=False)

    for anchor_idx, anchor in enumerate(X):
        
        negative_mask = labels != labels[anchor_idx]
        if len(labels[negative_mask]) < 1:
            #if there is no sample that has a different label from the anchor
            negative_mask = np.full(labels.shape, True)

        positive_mask = labels == labels[anchor_idx]
        if len(labels[positive_mask]) > 1:
            #if there is more than one sample with the same label, don't choose the anchor as the positive example
            positive_mask = np.concatenate((np.concatenate((positive_mask[:anchor_idx], [False])), positive_mask[anchor_idx+1:]))
            
        #calculate distance between anchor and positive/negative samples based on the chosen strategy
        if strategy == "random":
            dist_anchor_positive = dist_func(anchor, np.random.choice(X[positive_mask]))
            dist_anchor_negative = dist_func(anchor, np.random.choice(X[negative_mask]))
        elif strategy == "minmax":
            current = K[anchor_idx]
            dist_anchor_positive = current[positive_mask].min()
            dist_anchor_negative = current[negative_mask].max()
        
        sum = sum + max(dist_anchor_positive - dist_anchor_negative + alpha, 0)
    return sum