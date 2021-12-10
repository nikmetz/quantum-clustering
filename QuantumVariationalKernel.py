import pennylane as qml
import CostFunctions
from pennylane import numpy as np
from KernelKMeans import KernelKMeans
from sklearn import metrics
from Ansatz import ansatz2

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

    def train(self, X, Y=None, train_size=20, batch_size=5, val_size=5, epochs=500):
        opt = qml.GradientDescentOptimizer(0.05)

        num_samples = X.shape[0]
        if train_size+val_size > num_samples:
            raise ValueError("Provided {} samples, but required for train and validation set are {} samples.".format(num_samples, train_size+val_size))

        train_X, val_X = X[:train_size], X[train_size:train_size+val_size]
        if Y is not None:
            #Unsupervised -> no Y
            train_Y, val_Y = Y[:train_size], Y[train_size:train_size+val_size]

        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            perm = np.random.permutation(train_X.shape[0])
            batches = int(train_size/batch_size)
            parts = [perm[i::batches] for i in range(batches)]

            for idx, subset in enumerate(parts):
                if self.cost_func == "KTA-supervised":
                    cost = lambda _params: -qml.kernels.target_alignment(train_X[subset], train_Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params))
                    self.params, c = opt.step_and_cost(cost, self.params)
                    cost_val = qml.kernels.target_alignment(val_X, val_Y, self.kernel)
                    print("Step: {}, cost: {}, val_cost: {}".format(epoch*batches+idx, c, cost_val))
                elif self.cost_func == "triplet-loss-supervised":
                    cost = lambda _params: CostFunctions.triplet_loss(train_X[subset], train_Y[subset], 0.5, lambda x1, x2: self.kernel_with_params(x1, x2, _params))
                elif self.cost_func == "davies-bouldin":
                    kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                    cost = lambda _params: metrics.davies_bouldin_score(
                        train_X[subset],
                        kmeans.fit(train_X[subset], kernel=lambda x1, x2: self.kernel_with_params(x1, x2, _params)).labels_
                    )
                    self.params, c = opt.step_and_cost(cost, self.params)
                elif self.cost_func == "calinski-harabasz":
                    pass
                elif self.cost_func == "KTA-unsupervised":
                    pass
                elif self.cost_func == "triplet-loss-unsupervised":
                    pass
                else:
                    pass

            if (epoch + 1) % 100 == 0:
                km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                lab = km.fit(X, self.kernel).labels_
                davies = metrics.davies_bouldin_score(X, lab)
                calinski = metrics.calinski_harabasz_score(X, lab)
                print("Clustering test! Epoch {}: davies-bouldin: {}, calinski-harabasz: {}".format(epoch, davies, calinski))


if __name__ == '__main__':
    from datasets.iris import iris
    data = iris(train_size=50, test_size=0, shuffle=True)
    Y_new = np.array([y if y == 1 else -1 for y in data.train_target])

    wires = 4
    layers = 3

    init_params = random_params(num_wires=wires, num_layers=layers, params_per_wire=2)
    qvk = QuantumVariationalKernel(wires=wires, ansatz=ansatz2, init_params=init_params, cost="KTA-supervised")

    #qvk.train(X, Y_new)
    qvk.train(data.train_data, Y_new)