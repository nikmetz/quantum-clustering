import pennylane as qml
from qclustering.CostFunctions import get_cost_func
from pennylane import numpy as np

class QuantumVariationalKernel():
    def __init__(self, wires, ansatz, init_params, device, shots):
        self.device = qml.device(device, wires=wires, shots=shots)
        self.ansatz = ansatz
        self.qnode = qml.QNode(self.kernel_circuit, self.device)
        self.wires = wires
        self.params = init_params

    def kernel_circuit(self, x1, x2, params):
        self.ansatz.ansatz(x1, params)
        self.ansatz.adjoint_ansatz(x2, params)
        return qml.probs(wires=self.device.wires.tolist())

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def kernel_with_params(self, x1, x2, params):
        return self.qnode(x1, x2, params)[0]

    def distance(self, x1, x2):
        return 1 - self.kernel(x1, x2)

    def distance_with_params(self, x1, x2, params):
        return 1 - self.kernel_with_params(x1, x2, params)

    def train(
        self,
        data,
        batch_size = 5,
        epochs = 500,
        learning_rate = 0.2,
        learning_rate_decay = 0.0,
        optimizer_name = "GradientDescent",
        optimizer_params = {},
        cost_func_name = "KTA-supervised",
        cost_func_params = {},
        logging_obj = None
    ):

        train_X, train_Y = data.train_data, data.train_target
        val_X, val_Y = data.validation_data, data.validation_target
        test_X, test_Y = data.test_data, data.test_target

        n_clusters = np.unique(train_Y).shape[0]

        if logging_obj is not None:
            logging_obj.log_testing(0, self.kernel, n_clusters, test_X, test_Y, train_X, train_Y)

        for epoch in range(epochs):
            lrate = learning_rate * (1 / (1+learning_rate_decay*epoch))
            if optimizer_name == "GradientDescent":
                opt = qml.GradientDescentOptimizer(stepsize=lrate, **optimizer_params)
            elif optimizer_name == "Adam":
                opt = qml.AdamOptimizer(stepsize=lrate, **optimizer_params)
            elif optimizer_name == "Adagrad":
                opt = qml.AdagradOptimizer(stepsize=lrate, **optimizer_params)
            elif optimizer_name == "Momentum":
                opt = qml.MomentumOptimizer(stepsize=lrate, **optimizer_params)
            elif optimizer_name == "RMSProp":
                opt = qml.RMSPropOptimizer(stepsize=lrate, **optimizer_params)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            print("Epoch: {}, rate: {}".format(epoch, lrate))

            #TODO: is using a random partition here a good idea?
            perm = np.random.permutation(train_X.shape[0])
            batches = int(train_X.shape[0]/batch_size)
            parts = [perm[i::batches] for i in range(batches)]

            for idx, subset in enumerate(parts):
                cost_func = get_cost_func(cost_func_name, cost_func_params, self, n_clusters)

                self.params, c = opt.step_and_cost(lambda _params: cost_func(train_X[subset], train_Y[subset], _params), self.params)
                cost_val = cost_func(val_X, val_Y, self.params)
                print("Step: {}, cost: {}, val_cost: {}".format(epoch*batches+idx, c, cost_val))
                if logging_obj is not None:
                    logging_obj.log_training(epoch*batches+idx, c)
                    logging_obj.log_validation(epoch*batches+idx, cost_val)

            if logging_obj is not None:
                logging_obj.log_testing(epoch+1, self.kernel, n_clusters, test_X, test_Y, train_X, train_Y)

        if logging_obj is not None:
            logging_obj.finish()