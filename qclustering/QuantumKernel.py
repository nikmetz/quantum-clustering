import pennylane as qml
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding

n_qubits = 2
dev =  qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def angle_embedding(x1, x2):
    AngleEmbedding(x1, wires=range(n_qubits), rotation='X')
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits), rotation='X')
    return qml.probs(wires=dev.wires.tolist())

@qml.qnode(dev)
def amplitude_kernel(x1, x2):
    AmplitudeEmbedding(x1, wires=range(n_qubits), normalize=True)
    qml.adjoint(AmplitudeEmbedding)(x2, wires=range(n_qubits), normalize=True)
    return qml.probs(wires=dev.wires.tolist())

def kernel(x1, x2, kernel_type):
    if kernel_type == "angle":
        return angle_embedding(x1, x2)[0]
    elif kernel_type == "amplitude":
        return amplitude_kernel(x1, x2)[0]

def kernel_matrix(X, kernel_type):
    k = lambda x1, x2: kernel(x1, x2, kernel_type)
    #TODO: because of assume_normalized_kernel the kernel between identical datapoints is not calculated -> possibly needed when doing simulations with noise
    return qml.kernels.square_kernel_matrix(X, k, assume_normalized_kernel=True)