import autograd.numpy as np
import pennylane as qml

def get_cost_func(cost_func_name, cost_func_params, kernel_obj, n_clusters):
    if cost_func_name == "KTA-supervised":
        return lambda X, Y, params: -multiclass_target_alignment(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters, **cost_func_params)
    elif cost_func_name == "original-KTA-supervised":
        return lambda X, Y, params: -qml.kernels.target_alignment(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), **cost_func_params)
    elif cost_func_name == "kernel-penalty":
        return lambda X, Y, params: kernel_penalty(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), **cost_func_params)
    elif cost_func_name == "KTA-unsupervised":
        pass
    elif cost_func_name == "triplet-loss-supervised":
        return lambda X, Y, params: triplet_loss(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), **cost_func_params)
    elif cost_func_name == "triplet-loss-unsupervised":
        pass
    else:
        raise ValueError(f"Unknown cost function: {cost_func_name}")
    pass

def centered_kernel_mat(kernel_mat):
    n_samples = kernel_mat.shape[0]
    M = np.identity(n_samples) - np.full((n_samples, n_samples), 1/n_samples)
    return np.dot(np.dot(M, kernel_mat), M)

def multiclass_target_alignment(X, Y, kernel, num_classes, assume_normalized_kernel=False):
    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )
    num_samples = Y.shape[0]

    T = None
    for i in range(num_samples):
        n = [[1 if Y[i]==Y[j] else (-1/(num_classes-1)) for j in range(num_samples)]]
        if T is None:
            T = np.copy(n)
        else:
            T = np.concatenate((T, n), axis=0)

    return qml.math.frobenius_inner_product(centered_kernel_mat(K), centered_kernel_mat(T), normalize=True)

def kernel_penalty(X, Y, kernel, strategy="mean", inner_penalty=1, inter_penalty=1):
    K = qml.kernels.square_kernel_matrix(X, kernel, assume_normalized_kernel=False)
    inner = 0.0
    inter = 0.0
    inner_count = 0
    inter_count = 0

    for x in range(len(X)):
        for y in range(x, len(X), 1):
            if Y[x] == Y[y]:
                inner += 1 - K[x][y]
                inner_count += 1
            else:
                inter += K[x][y]
                inter_count += 1
    if strategy == "mean":
        if inner_count > 0:
            inner = inner/inner_count
        if inter_count > 0:
            inter = inter/inter_count

    return inner_penalty * inner + inter_penalty * inter

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
            pos_idx = np.random.randint(X[positive_mask].shape[0], size=1)[0]
            neg_idx = np.random.randint(X[negative_mask].shape[0], size=1)[0]
            dist_anchor_positive = dist_func(anchor, X[positive_mask][pos_idx])
            dist_anchor_negative = dist_func(anchor, X[negative_mask][neg_idx])
        elif strategy == "minmax":
            current = K[anchor_idx]
            dist_anchor_positive = current[positive_mask].max()
            dist_anchor_negative = current[negative_mask].min()
        
        # dist_func is a kernel function, which is a similarity function, but triplet loss uses distances
        dist_anchor_positive = 1 - dist_anchor_positive
        dist_anchor_negative = 1 - dist_anchor_negative
        sum = sum + max(dist_anchor_positive - dist_anchor_negative + alpha, 0)
    return sum