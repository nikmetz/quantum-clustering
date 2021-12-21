import pennylane as qml
import qclustering.CostFunctions as CostFunctions
from pennylane import numpy as np
from qclustering.KernelKMeans import KernelKMeans
from sklearn import metrics
from qclustering.utils import simple_plot

class QuantumVariationalKernel():
    def __init__(self, wires, ansatz, init_params, cost_func, device, shots):
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

    def train(self, data, **kwargs):
        batch_size = kwargs.get('batch_size', 5)
        epochs = kwargs.get('epochs', 500)
        learning_rate = kwargs.get('learning_rate', 0.2)
        learning_rate_decay = kwargs.get('learning_rate_decay', 0.01)
        clustering_interval = kwargs.get("clustering_interval", 100)
        opt_name = kwargs.get("optimizer", "GradientDescent")
        num_classes = kwargs.get("num_classes", 2)
        path = kwargs.get("path")

        train_X, train_Y = data.train_data, data.train_target
        val_X, val_Y = data.validation_data, data.validation_target
        test_X, test_Y = data.test_data, data.test_target

        val_values = []
        clustering_values = []

        for epoch in range(epochs):
            lrate = learning_rate * (1 / (1+learning_rate_decay*epoch))
            if opt_name == "GradientDescent":
                opt = qml.GradientDescentOptimizer(lrate)
            print("Epoch: {}, rate: {}".format(epoch, lrate))

            #TODO: is using a random partition here a good idea?
            perm = np.random.permutation(train_X.shape[0])
            batches = int(train_X.shape[0]/batch_size)
            parts = [perm[i::batches] for i in range(batches)]

            for idx, subset in enumerate(parts):
                #TODO: should probably find a better solution for this mess
                if self.cost_func == "KTA-supervised":
                    cost = lambda _params: -CostFunctions.multiclass_target_alignment(train_X[subset], train_Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params), num_classes)
                    self.params, c = opt.step_and_cost(cost, self.params)
                    cost_val = CostFunctions.multiclass_target_alignment(val_X, val_Y, self.kernel, num_classes)
                    print("Step: {}, cost: {}, val_cost: {}".format(epoch*batches+idx, c, cost_val))
                    val_values.append((epoch*batches+idx, cost_val))

                elif self.cost_func == "triplet-loss-supervised":
                    cost = lambda _params: CostFunctions.triplet_loss(train_X[subset], train_Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params), **kwargs.get("cost_func_params", "{}"))
                    self.params, c = opt.step_and_cost(cost, self.params)
                    cost_val = CostFunctions.triplet_loss(val_X, val_Y, self.kernel, **kwargs.get("cost_func_params", "{}"))
                    print("Step: {}, cost: {}, val_cost: {}".format(epoch*batches+idx, c, cost_val))
                    val_values.append((epoch*batches+idx, cost_val))

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

            if (epoch + 1) % clustering_interval == 0:
                km = KernelKMeans(n_clusters=num_classes, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                lab = km.fit(test_X, self.kernel).labels_

                davies = metrics.davies_bouldin_score(test_X, lab)
                calinski = metrics.calinski_harabasz_score(test_X, lab)
                if test_Y is not None:
                    rand_index = metrics.adjusted_rand_score(test_Y, lab)
                    mutual_info = metrics.adjusted_mutual_info_score(test_Y, lab)
                else:
                    rand_index = 0
                    mutual_info = 0

                print("Clustering test! Epoch {}: davies-bouldin: {}, calinski-harabasz: {}, rand: {}, info: {}".format(epoch, davies, calinski, rand_index, mutual_info))
                clustering_values.append((epoch*batches, davies, calinski, rand_index, mutual_info))

        simple_plot([x[0] for x in val_values], [x[1] for x in val_values], "Steps", "Cost", path+"loss.png", [-1, 1])

        xs = [x[0] for x in clustering_values]
        simple_plot(xs, [x[1] for x in clustering_values], "Steps", "Davies", path+"davies.png")
        simple_plot(xs, [x[2] for x in clustering_values], "Steps", "Calinski", path+"calinski.png")
        simple_plot(xs, [x[3] for x in clustering_values], "Steps", "Rand Index", path+"rand_index.png")
        simple_plot(xs, [x[4] for x in clustering_values], "Steps", "Mutual Info", path+"mutual_info.png")