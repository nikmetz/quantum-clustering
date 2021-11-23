import pennylane as qml
from pennylane import numpy as np
from KernelKMeans import KernelKMeans
from utils import visualize
from sklearn import metrics
from Ansatz import ansatz

def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires))

class QuantumVariationalKernel():
    def __init__(self, wires, ansatz, init_params, device="default.qubit", shots=None):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.qnode = qml.QNode(self.kernel_circuit, self.device, diff_method="parameter-shift")
        self.wires = self.device.wires.tolist()
        self.ansatz = ansatz
        self.params = init_params
        self.adjoint_ansatz = qml.adjoint(self.ansatz)

    def kernel_circuit(self, x1, x2, params):
        self.ansatz(x1, params, wires=self.wires)
        self.adjoint_ansatz(x2, params, wires=self.wires)
        return qml.probs(wires=self.wires)

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return self.qnode(x1, x2, params)[0]

    def train(self, X):
        opt = qml.GradientDescentOptimizer(0.2)

        subset = np.random.choice(list(range(len(X))), 10)
        for i in range(5):
            kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
            # Define the cost function for optimization
            cost = lambda _params: metrics.davies_bouldin_score(
                X[subset],
                kmeans.fit(X, kernel=lambda x1, x2: self.kernel_with_params(x1, x2, _params)).labels_
            )
            # Optimization step
            self.params, c = opt.step_and_cost(cost, self.params)

            # Report the alignment on the full dataset every 50 steps.
            if (i + 1) % 50 == 0:
                score = metrics.davies_bouldin_score(X, kmeans.fit(X, lambda x1, x2: self.kernel(x1, x2)).labels_)
                print(f"Step {i+1} - Score = {score:.3f}")

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=10, centers=2, random_state=0)

    init_params = random_params(num_wires=5, num_layers=6)
    qvk = QuantumVariationalKernel(5, ansatz=ansatz, init_params=init_params)
    #init_kernel = lambda x1, x2: qvk.kernel_with_params(x1, x2, init_params)
    #K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)
    #km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=init_kernel)

    #result = km.fit(X).labels_

    #visualize(X, y, result)
    #print(qvk.target_alignment(X, y, init_kernel, assume_normalized_kernel=True))
    qvk.train(X)