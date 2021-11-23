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
    def __init__(self, wires, ansatz, init_params, cost, device="default.qubit", shots=None):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.ansatz = ansatz
        self.qnode = qml.QNode(self.kernel_circuit, self.device)
        self.wires = self.device.wires.tolist()
        self.params = init_params
        self.adjoint_ansatz = qml.adjoint(self.ansatz)
        self.cost_func = cost

    def kernel_circuit(self, x1, x2, params):
        self.ansatz(x1, params, wires=self.wires)
        self.adjoint_ansatz(x2, params, wires=self.wires)
        return qml.probs(wires=self.wires)

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return self.qnode(x1, x2, params)[0]

    def target_alignment(
        self,
        X,
        Y,
        kernel,
        assume_normalized_kernel=False,
        rescale_class_labels=True,
    ):
        """Kernel-target alignment between kernel and labels."""

        K = qml.kernels.square_kernel_matrix(
            X,
            kernel,
            assume_normalized_kernel=assume_normalized_kernel,
        )

        if rescale_class_labels:
            nplus = np.count_nonzero(np.array(Y) == 1)
            nminus = len(Y) - nplus
            _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
        else:
            _Y = np.array(Y)

        T = np.outer(_Y, _Y)
        inner_product = np.sum(K * T)
        norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
        inner_product = inner_product / norm

        return inner_product

    def train(self, X, Y=None):
        opt = qml.GradientDescentOptimizer(0.2)

        for i in range(100):
            subset = np.random.choice(list(range(len(X))), 4)
            if self.cost_func == "KTA":
                cost = lambda _params: -self.target_alignment(X[subset], Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params),assume_normalized_kernel=True)
            else:
                kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                # Define the cost function for optimization
                cost = lambda _params: metrics.davies_bouldin_score(
                    X[subset],
                    kmeans.fit(X, kernel=lambda x1, x2: self.kernel_with_params(x1, x2, _params)).labels_
                )
            # Optimization step
            self.params= opt.step(cost, self.params)

            if (i + 1) % 1 == 0:
                current_alignment = self.target_alignment(
                    X,
                    Y,
                    lambda x1, x2: self.kernel_with_params(x1, x2, self.params),
                    assume_normalized_kernel=True,
                )
                print(f"Step {i+1} - Alignment = {current_alignment:.3f}")

if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=10, centers=2, random_state=0)
    Y_new = np.array([y if y == 1 else -1 for y in Y])

    init_params = random_params(num_wires=5, num_layers=6)
    qvk = QuantumVariationalKernel(wires=5, ansatz=ansatz, init_params=init_params, cost="KTA")
    #init_kernel = lambda x1, x2: qvk.kernel_with_params(x1, x2, init_params)
    #K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)
    #km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=init_kernel)

    #result = km.fit(X).labels_

    #visualize(X, y, result)
    #print(qvk.target_alignment(X, y, init_kernel, assume_normalized_kernel=True))
    qvk.train(X, Y_new)