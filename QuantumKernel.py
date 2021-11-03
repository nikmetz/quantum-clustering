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

def angle_kernel(x1, x2):
    return kernel("angle", x1, x2)

def kernel(kernel_type, x1, x2):
    if kernel_type == "angle":
        return angle_embedding(x1, x2)[0]
    elif kernel_type == "amplitude":
        return amplitude_kernel(x1, x2)[0]

def kernel_matrix(kernel_type, X, Y=None):
    pass