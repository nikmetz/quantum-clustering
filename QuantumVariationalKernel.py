import pennylane as qml
import CostFunctions
from pennylane import numpy as np
from KernelKMeans import KernelKMeans
from sklearn import metrics
from Ansatz import ansatz2
from utils import simple_plot

def fixed_value_params(value, num_wires, num_layers, params_per_wire):
    return np.full((num_layers, num_wires, params_per_wire), value)

def random_params(num_wires, num_layers, params_per_wire):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, num_wires, params_per_wire))

class QuantumVariationalKernel():
    def __init__(self, wires, ansatz, init_params, cost_func, device="default.qubit", shots=None):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.ansatz = ansatz
        self.qnode = qml.QNode(self.kernel_circuit, self.device)
        self.wires = wires
        self.params = init_params
        self.adjoint_ansatz = qml.adjoint(self.ansatz)
        self.cost_func = cost_func

    def kernel_circuit(self, x1, x2, params):
        self.ansatz(x1, params, wires=self.wires)
        self.adjoint_ansatz(x2, params, wires=self.wires)
        return qml.probs(wires=self.device.wires.tolist())

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return 1 - self.qnode(x1, x2, params)[0]

    def train(self, X, Y=None, **kwargs):
        train_size = kwargs.get('train_size', 20)
        batch_size = kwargs.get('batch_size', 5)
        val_size = kwargs.get('val_size', 5)
        epochs = kwargs.get('epochs', 10)
        learning_rate = kwargs.get('learning_rate', 0.2)
        learning_rate_decay = kwargs.get('learning_rate_decay', 0.01)

        num_samples = X.shape[0]
        if train_size+val_size > num_samples:
            raise ValueError("Provided {} samples, but required for train and validation set are {} samples.".format(num_samples, train_size+val_size))

        train_X, val_X = X[:train_size], X[train_size:train_size+val_size]
        if Y is not None:
            #Semi-supervised -> Y values
            train_Y, val_Y = Y[:train_size], Y[train_size:train_size+val_size]

        val_values = []
        clustering_values = []

        for epoch in range(epochs):
            lrate = learning_rate * (1 / (1+learning_rate_decay*epoch))
            opt = qml.GradientDescentOptimizer(lrate)
            print("Epoch: {}, rate: {}".format(epoch, lrate))

            #TODO: is using a random partition here a good idea?
            perm = np.random.permutation(train_X.shape[0])
            batches = int(train_size/batch_size)
            parts = [perm[i::batches] for i in range(batches)]

            for idx, subset in enumerate(parts):
                #TODO: should probably find a better solution for this mess
                if self.cost_func == "KTA-supervised":
                    cost = lambda _params: -qml.kernels.target_alignment(train_X[subset], train_Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params))
                    self.params, c = opt.step_and_cost(cost, self.params)
                    cost_val = qml.kernels.target_alignment(val_X, val_Y, self.kernel)
                    print("Step: {}, cost: {}, val_cost: {}".format(epoch*batches+idx, c, cost_val))
                    val_values.append((epoch*batches+idx, cost_val))
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

            if (epoch + 1) % 1000 == 0:
                km = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                lab = km.fit(X, self.kernel).labels_
                davies = metrics.davies_bouldin_score(X, lab)
                calinski = metrics.calinski_harabasz_score(X, lab)
                if Y is not None:
                    rand_index = metrics.adjusted_rand_score(Y, lab)
                    mutual_info = metrics.adjusted_mutual_info_score(Y, lab)
                else:
                    rand_index = 0
                    mutual_info = 0
                print("Clustering test! Epoch {}: davies-bouldin: {}, calinski-harabasz: {}, rand: {}, info: {}".format(epoch, davies, calinski, rand_index, mutual_info))
                clustering_values.append((epoch*batches, davies, calinski, rand_index, mutual_info))

        simple_plot([x[0] for x in val_values], [x[1] for x in val_values], "Steps", "KTA", "kta.png", [-1, 1])

        xs = [x[0] for x in clustering_values]
        simple_plot(xs, [x[1] for x in clustering_values], "Steps", "Davies", "davies.png")
        simple_plot(xs, [x[2] for x in clustering_values], "Steps", "Calinski", "calinski.png")
        simple_plot(xs, [x[3] for x in clustering_values], "Steps", "Rand Index", "rand_index.png")
        simple_plot(xs, [x[4] for x in clustering_values], "Steps", "Mutual Info", "mutual_info.png")

if __name__ == '__main__':
    from datasets.iris import iris
    np.random.seed(42)
    data = iris(train_size=25, test_size=0, shuffle=True)
    Y_new = np.array([y if y == 1 else -1 for y in data.train_target])

    wires = 4
    layers = 3

    init_params = random_params(num_wires=wires, num_layers=layers, params_per_wire=2)
    qvk = QuantumVariationalKernel(wires=wires, ansatz=ansatz2, init_params=init_params, cost_func="KTA-supervised")

    #qvk.train(X, Y_new)
    qvk.train(data.train_data, Y_new)