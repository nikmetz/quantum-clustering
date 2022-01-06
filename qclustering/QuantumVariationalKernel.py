import pennylane as qml
import qclustering.CostFunctions as CostFunctions
from pennylane import numpy as np
from qclustering.KernelKMeans import KernelKMeans
from sklearn import metrics
from qclustering.Logging import Logging, get_cluster_result

class QuantumVariationalKernel():
    def __init__(self, wires, ansatz, init_params, cost_func, device, shots):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.ansatz = ansatz
        self.qnode = qml.QNode(self.kernel_circuit, self.device)
        self.wires = wires
        self.params = init_params
        self.adjoint_ansatz = qml.adjoint(self.ansatz)
        self.cost_func_name = cost_func

    def kernel_circuit(self, x1, x2, params):
        self.ansatz(x1, params, wires=self.wires)
        self.adjoint_ansatz(x2, params, wires=self.wires)
        return qml.probs(wires=self.device.wires.tolist())

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return self.qnode(x1, x2, params)[0]

    def distance(self, x1, x2):
        return 1 - self.kernel(x1, x2)

    def distance_with_params(self, x1, x2, params):
        return 1 - self.kernel_with_params(x1, x2, params)

    def train(self, data, **kwargs):
        batch_size = kwargs.get('batch_size', 5)
        epochs = kwargs.get('epochs', 500)
        learning_rate = kwargs.get('learning_rate', 0.2)
        learning_rate_decay = kwargs.get('learning_rate_decay', 0.01)
        clustering_interval = kwargs.get("clustering_interval", 100)
        opt_name = kwargs.get("optimizer", "GradientDescent")
        num_classes = max(len(kwargs.get("dataset_params", {}).get("classes", [])), 2)
        path = kwargs.get("path")

        train_X, train_Y = data.train_data, data.train_target
        val_X, val_Y = data.validation_data, data.validation_target
        test_X, test_Y = data.test_data, data.test_target

        logger = Logging()

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
                if self.cost_func_name == "KTA-supervised":
                    cost_func = lambda _params: -qml.kernels.target_alignment(train_X[subset], train_Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params), assume_normalized_kernel=False, rescale_class_labels=True)
                    cost_val_func = lambda k: -qml.kernels.target_alignment(val_X, val_Y, k, assume_normalized_kernel=False, rescale_class_labels=True)

                elif self.cost_func_name == "triplet-loss-supervised":
                    cost_func = lambda _params: CostFunctions.triplet_loss(train_X[subset], train_Y[subset], lambda x1, x2: self.kernel_with_params(x1, x2, _params), **kwargs.get("cost_func_params", "{}"))
                    cost_val_func = lambda k: CostFunctions.triplet_loss(val_X, val_Y, k, **kwargs.get("cost_func_params", "{}"))

                elif self.cost_func_name == "davies-bouldin":
                    kmeans = KernelKMeans(n_clusters=2, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                    cost = lambda _params: metrics.davies_bouldin_score(
                        train_X[subset],
                        kmeans.fit(train_X[subset], kernel=lambda x1, x2: self.kernel_with_params(x1, x2, _params)).labels_
                    )
                    self.params, c = opt.step_and_cost(cost, self.params)
                    
                elif self.cost_func_name == "calinski-harabasz":
                    pass
                elif self.cost_func_name == "KTA-unsupervised":
                    pass
                elif self.cost_func_name == "triplet-loss-unsupervised":
                    pass
                else:
                    raise ValueError(f"Unknown cost function: {self.cost_func_name}")

                self.params, c = opt.step_and_cost(cost_func, self.params)
                cost_val = cost_val_func(self.kernel)
                print("Step: {}, cost: {}, val_cost: {}".format(epoch*batches+idx, c, cost_val))
                logger.log_training(epoch*batches+idx, c, cost_val)

            if (epoch + 1) % clustering_interval == 0:
                km = KernelKMeans(n_clusters=num_classes, max_iter=100, random_state=0, verbose=1, kernel=self.kernel)
                lab = km.fit(test_X, self.kernel).labels_
                print(lab)
                print(test_Y)

                result = get_cluster_result(test_X, test_Y, lab)
                logger.log_clustering(epoch*batches, result)

        logger.generate_imgs(path)
        logger.write_csv(path)