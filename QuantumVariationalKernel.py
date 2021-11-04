import pennylane as qml
from pennylane import numpy as np
from KernelKMeans import KernelKMeans
from utils import visualize

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires))

class QuantumVariationalKernel():
    def __init__(self, wires, layer, init_params, device="default.qubit", shots=None):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.qnode = qml.QNode(self.kernel_circuit, self.device)
        self.wires = self.device.wires.tolist()
        self.layer = layer
        self.params = init_params
        self.adjoint_ansatz = qml.adjoint(self.ansatz)

    def ansatz(self, x, params, wires):
        for j, layer_params in enumerate(params):
            self.layer(x, layer_params, wires, i0=j * len(wires))

    def kernel_circuit(self, x1, x2, params):
        self.ansatz(x1, params, wires=self.wires)
        self.adjoint_ansatz(x2, params, wires=self.wires)
        return qml.probs(wires=self.wires)

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return self.qnode(x1, x2, self.params)[0]

    def train():
        #TODO
        pass

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, centers=2, random_state=0)

    qvk = QuantumVariationalKernel(5, layer=layer)
    init_params = random_params(num_wires=5, num_layers=6)
    init_kernel = lambda x1, x2: qvk.kernel(x1, x2, init_params)
    #K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)
    km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=init_kernel)
    
    result = km.fit(X).labels_

    visualize(X, y, result)