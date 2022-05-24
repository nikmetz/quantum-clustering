import autograd.numpy as np
import pennylane as qml
from qclustering.Clustering import clustering
from itertools import permutations
import copy
from sklearn import metrics
from qclustering.Clustering import clustering
from qclustering.KernelMatrix import square_postprocessing

def get_cost_func(X, Y, cost_func_name, cost_func_params, kernel_obj, n_clusters):
    if cost_func_name == "KTA":
        return lambda params: -multiclass_target_alignment(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters, **cost_func_params)
    elif cost_func_name == "original-KTA":
        return lambda params: -qml.kernels.target_alignment(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), **cost_func_params)
    elif cost_func_name == "hilbert-schmidt":
        return lambda params: hilbert_schmidt(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params))
    elif cost_func_name == "clustering-risk":
        return lambda params: clustering_risk(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters)
    elif cost_func_name == "triplet-loss":
        return lambda params: triplet_loss(X, Y, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), **cost_func_params)
    elif cost_func_name == "rand-score":
        return lambda params: -metrics.rand_score(Y, clustering("kmeans", {}, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters, X))
    elif cost_func_name == "adjusted-rand-score":
        return lambda params: -metrics.adjusted_rand_score(Y, clustering("kmeans", {}, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters, X))
    elif cost_func_name == "davies-bouldin":
        return lambda params: metric_cost_func(metrics.davies_bouldin_score, X, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters)
    elif cost_func_name == "calinski-harabasz":
        return lambda params: -metric_cost_func(metrics.calinski_harabasz_score, X, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters)
    elif cost_func_name == "test":
        return test(X, kernel_obj, n_clusters)
    else:
        raise ValueError(f"Unknown cost function: {cost_func_name}")
    pass

def get_above_thresh(A, thresh):
    b = []
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i != j and A[i][j] >= thresh:
                if i not in b:
                    b.append(i)
                if j not in b:
                    b.append(j)
    b.sort()
    return b

def count_above_thresh(A, count, starting_thresh):
    thresh = starting_thresh
    c = get_above_thresh(A, thresh)
    while len(c) < count:
        thresh -= 0.05
        c = get_above_thresh(A, thresh)
    return c

def test(X, kernel_obj, n_clusters):
    K = qml.kernels.square_kernel_matrix(
        X,
        kernel_obj.kernel,
        assume_normalized_kernel=False,
    )
    c = count_above_thresh(K, 5, 0.9)
    reduced_X = X[c]
    reduced_K = K[c,:][:,c]

    true_labels = clustering("kmeans", {}, "precomputed", n_clusters, reduced_K)

    return lambda params: -multiclass_target_alignment(reduced_X, true_labels, lambda x1, x2: kernel_obj.kernel_with_params(x1, x2, params), n_clusters)

def metric_cost_func(metric_func, X, kernel, n_clusters):
    labels = clustering("kmeans", {}, kernel, n_clusters, X)
    if len(np.unique(labels)) != n_clusters:
        return 0.
    else:
        return metric_func(X, labels)

def centered_kernel_mat(kernel_mat):
    n_samples = kernel_mat.shape[0]
    M = np.identity(n_samples) - np.full((n_samples, n_samples), 1/n_samples)
    return np.dot(np.dot(M, kernel_mat), M)

def multiclass_target_alignment(X, Y, kernel, num_classes, assume_normalized_kernel=False):
    if kernel == "precomputed":
        K = X
    else:
        K = qml.kernels.square_kernel_matrix(
            X,
            kernel,
            assume_normalized_kernel=assume_normalized_kernel,
        )
        K = square_postprocessing(K)
    num_samples = Y.shape[0]

    T = None
    for i in range(num_samples):
        n = [[1 if Y[i]==Y[j] else (-1/(num_classes-1)) for j in range(num_samples)]]
        if T is None:
            T = np.copy(n)
        else:
            T = np.concatenate((T, n), axis=0)

    return qml.math.frobenius_inner_product(centered_kernel_mat(K), centered_kernel_mat(T), normalize=True)

def hilbert_schmidt(X, Y, kernel):
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
    if inner_count > 0:
            inner = inner/inner_count
    if inter_count > 0:
        inter = inter/inter_count
    
    return 1 - 0.5 * (inner - (2 * inter))


def triplet_loss(X, labels, kernel, strategy="random", alpha=1):

    sum = 0.0
    if strategy == "minmax":
        #only needed for minmax strategy, not random
        K = qml.kernels.square_kernel_matrix(X, kernel, assume_normalized_kernel=False)
        K = square_postprocessing(K)

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
            k_anchor_positive = kernel(anchor, X[positive_mask][pos_idx])
            k_anchor_negative = kernel(anchor, X[negative_mask][neg_idx])
        elif strategy == "minmax":
            current = K[anchor_idx]
            k_anchor_positive = current[positive_mask].max()
            k_anchor_negative = current[negative_mask].min()
        
        sum = sum + max(alpha - k_anchor_positive + k_anchor_negative, 0)
    return sum

def clustering_risk(X, labels, kernel, n_clusters):
    K = qml.kernels.square_kernel_matrix(X, kernel, assume_normalized_kernel=False)
    pred_labels = clustering("kmeans", {}, kernel, n_clusters, X)

    cost = None

    unique_labels = np.unique(labels)
    permut = set(permutations(unique_labels))
    unique_pred_labels = np.unique(pred_labels)

    for x in permut:
        a = copy.deepcopy(pred_labels)
        masks = []
        for l in unique_pred_labels:
            masks.append(a == l)
        for idx, y in enumerate(masks):
            a[y] = x[idx]

        su = 0.0
        for idx, x in enumerate(labels):
            su += x * a[idx]

        c = 0.5 - (1/(2*len(labels))) * su

        if cost is None:
            cost = c
        else:
            cost = min(cost, c)

    return cost

        