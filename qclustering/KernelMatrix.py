import pennylane as qml
import multiprocessing
from pennylane import numpy as np

postp_functions = []
postp_parameters = []

processing_pool = None

def set_processing_pool(process_count):
    global processing_pool
    processing_pool = multiprocessing.Pool(processes=process_count)

def __kernel_help(i, j, x, y, kernel):
    return (i, j, kernel(x, y))

def parallel_square_kernel_matrix(X, kernel, assume_normalized_kernel=False):
    if processing_pool is None:
        return qml.kernels.square_kernel_matrix(X, kernel)

    N = len(X)
    matrix = [0] * N ** 2
    jobs = []
    for i in range(N):
        for j in range(i, N):
            if assume_normalized_kernel and i == j:
                matrix[N * i + j] = 1.0
            else:
                jobs.append((i, j, X[i], X[j], kernel))

    results = processing_pool.starmap(__kernel_help, jobs)
    for x in results:
        matrix[N * x[0] + x[1]] = x[2]
        matrix[N * x[1] + x[0]] = x[2]

    return np.array(matrix).reshape((N, N))

def parallel_kernel_matrix(X1, X2, kernel):
    if processing_pool is None:
        return qml.kernels.kernel_matrix(X1, X2, kernel)
        
    N = len(X1)
    M = len(X2)

    matrix = [0] * N * M
    for i in range(N):
        for j in range(M):
            matrix[M * i + j] = (X1[i], X2[j])
    matrix = processing_pool.starmap(kernel, matrix)

    return np.array(matrix).reshape((N, M))

def set_postprocessing_parameters(function_names, parameters):
    global postp_functions
    global postp_parameters

    for f,p in zip(function_names, parameters):
        if hasattr(qml.kernels, f):
            postp_functions.append(getattr(qml.kernels, f))
        else:
            raise ValueError(f"Unknown postprocessing function name: {f}")

        postp_parameters.append(p)
        
        
def square_postprocessing(mat):
    processed_mat = mat
    for func, p in zip(postp_functions, postp_parameters):
        processed_mat = func(processed_mat, **p)

    return processed_mat