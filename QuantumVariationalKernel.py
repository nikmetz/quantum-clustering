import pennylane as qml
from pennylane import numpy as np
from KernelKMeans import KernelKMeans
from sklearn import metrics
from Ansatz import ansatz2
import random

def random_params(num_wires, num_layers, params_per_wire):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, num_wires, params_per_wire))

class QuantumVariationalKernel():
    def __init__(self, wires, ansatz, init_params, cost, device="default.qubit", shots=None):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.ansatz = ansatz
        self.qnode = qml.QNode(self.kernel_circuit, self.device)
        self.wires = wires
        self.params = init_params
        self.adjoint_ansatz = qml.adjoint(self.ansatz)
        self.cost_func = cost

    def kernel_circuit(self, x1, x2, params):
        self.ansatz(x1, params, wires=self.wires)
        self.adjoint_ansatz(x2, params, wires=self.wires)
        return qml.probs(wires=self.device.wires.tolist())

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return 1-self.qnode(x1, x2, params)[0]

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

    def triplet_loss(self, X, labels, num_samples, alpha, params):
        sum = 0.0
        for i in range(num_samples):
            anchor_index = random.randrange(labels.shape[0])
            anchor = X[anchor_index]
            negative = random.choice(X[labels != labels[anchor_index]])
            positive_mask = labels == labels[anchor_index]
            #positive_mask = vector_change(positive_mask, False, anchor_index)
            positive = random.choice(X[positive_mask])
            sum = sum + max(self.kernel_with_params(anchor, positive, params) - self.kernel_with_params(anchor, negative, params) + alpha, 0)
        return sum

    def train(self, X, Y=None):
        opt = qml.GradientDescentOptimizer(0.2)

        for i in range(1000):
            print(i)
            subset = np.random.choice(list(range(len(X))), 5)
            if self.cost_func == "KTA":
                #Semi-supervised learning with the kernel target alignment
                cost = lambda _params: -qml.kernels.target_alignment(X[subset], Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params),assume_normalized_kernel=True)
            elif self.cost_func == "daviesbouldin":
                kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                cost = lambda _params: metrics.davies_bouldin_score(
                    X[subset],
                    kmeans.fit(X[subset], kernel=lambda x1, x2: self.kernel_with_params(x1, x2, _params)).labels_
                )
            elif self.cost_func == "tripletloss":
                kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                la = kmeans.fit(X[subset], self.kernel).labels_

                cost = lambda _params: self.triplet_loss(X[subset], la, 1, 0.5, _params)
            else:
                kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)

                la = kmeans.fit(X[subset], self.kernel).labels_
                la = np.array([y if y == 1 else -1 for y in la])
                cost = lambda _params: -qml.kernels.target_alignment(
                    X[subset],
                    la,
                    lambda x1, x2: self.kernel_with_params(x1, x2, _params),
                    assume_normalized_kernel=True,
                )

            # Optimization step
            self.params= opt.step(cost, self.params)

            # Output the current quality
            if (i + 1) % 50 == 0:
                km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                lab = km.fit(X, self.kernel).labels_
                print("{}, {}".format(lab, metrics.davies_bouldin_score(X, lab)))

    def semi_supervised(self, X, Y, val_size=5,epochs=500):
        opt = qml.GradientDescentOptimizer(0.01)

        num_samples = X.shape[0]
        train_X, train_Y = X[:num_samples-val_size], Y[:num_samples-val_size]
        val_X, val_Y = X[num_samples-val_size:], Y[num_samples-val_size:]

        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            perm = np.random.permutation(train_X.shape[0])
            parts = [perm[i::1] for i in range(1)]
            for idx, val in enumerate(parts):
                cost = lambda _params: -qml.kernels.target_alignment(train_X[val], train_Y[val], lambda x1, x2: self.kernel_with_params(x1, x2, _params))
                self.params, c = opt.step_and_cost(cost, self.params)
                c_val = qml.kernels.target_alignment(val_X, val_Y, self.kernel)
                print("Step: {}, cost: {}, val_cost: {}".format((epoch*1)+(idx), -c, c_val))
            if (epoch + 1) % 100 == 0:
                km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                lab = km.fit(X, self.kernel).labels_
                print("Epoch {}: {}".format(epoch, metrics.davies_bouldin_score(X, lab)))


if __name__ == '__main__':
    from datasets.iris import iris
    data = iris(train_size=55, test_size=0, shuffle=True)
    Y_new = np.array([y if y == 1 else -1 for y in data.train_target])

    wires = 4
    layers = 3

    init_params = random_params(num_wires=wires, num_layers=layers, params_per_wire=2)
    qvk = QuantumVariationalKernel(wires=wires, ansatz=ansatz2, init_params=init_params, cost="tripletloss")

    #qvk.train(X, Y_new)
    qvk.semi_supervised(data.train_data, Y_new)