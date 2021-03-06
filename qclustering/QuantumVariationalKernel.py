from distutils.command.build import build
import pennylane as qml
from qclustering.CostFunctions import get_cost_func
from qclustering.utils import build_noise_model_qiskit, build_noise_model_qml
from pennylane import numpy as np
from qclustering.NoiseExtension import NoiseExtension

class QuantumVariationalKernel():
    def __init__(self, ansatz, init_params, device, logging_obj = None):
        self.ansatz = ansatz
        self.qnode = qml.QNode(self.kernel_circuit, device)
        self.wires = device.wires.tolist()
        self.params = init_params
        self.logging_obj = logging_obj

    def kernel_circuit(self, x1, x2, params):
        pass

    def kernel_with_params(self, x1, x2, params):
        pass

    def kernel(self, x1, x2):
        return self.kernel_with_params(x1, x2, self.params)

    def distance(self, x1, x2):
        return 1 - self.kernel(x1, x2)

    def distance_with_params(self, x1, x2, params):
        return 1 - self.kernel_with_params(x1, x2, params)

    def train(
        self,
        data,
        batch_size = 5,
        num_batches = None,
        epochs = 500,
        learning_rate = 0.2,
        learning_rate_decay = 0.0,
        optimizer_name = "GradientDescent",
        optimizer_params = {},
        cost_func_name = "KTA-supervised",
        cost_func_params = {},
    ):

        train_X, train_Y = data.train_data, data.train_target
        val_X, val_Y = data.validation_data, data.validation_target
        test_X, test_Y = data.test_data, data.test_target

        n_clusters = np.unique(train_Y).shape[0]

        if num_batches is not None:
            batch_size = int(train_X.shape[0] / num_batches)

        if self.logging_obj is not None:
            self.logging_obj.log_train_clustering(0, self.kernel, n_clusters, train_X, train_Y)
            self.logging_obj.log_test_clustering(0, self.kernel, n_clusters, test_X, test_Y, train_X, train_Y)
            self.logging_obj.log_weights(0, self.params)

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
            parts = [perm[i:i+batch_size] for i in range(0, len(perm), batch_size)]

            for idx, subset in enumerate(parts):
                cost_func = get_cost_func(train_X[subset], train_Y[subset], cost_func_name, cost_func_params, self, n_clusters)
                self.params, c = opt.step_and_cost(cost_func, self.params)

                cost_func = get_cost_func(val_X, val_Y, cost_func_name, cost_func_params, self, n_clusters)
                cost_val = cost_func(self.params)

                step_num = epoch * len(parts) + idx
                print("Step: {}, cost: {}, val_cost: {}".format(step_num, c, cost_val))
                if self.logging_obj is not None:
                    self.logging_obj.log_train_cost(epoch*step_num+idx, c)
                    self.logging_obj.log_val_cost(epoch*step_num+idx, cost_val)
                    self.logging_obj.log_weights(epoch*step_num+idx, self.params)

            if self.logging_obj is not None:
                self.logging_obj.log_train_clustering(epoch+1, self.kernel, n_clusters, train_X, train_Y)
                self.logging_obj.log_test_clustering(epoch+1, self.kernel, n_clusters, test_X, test_Y, train_X, train_Y)

        if self.logging_obj is not None:
            self.logging_obj.finish()

class OverlapQuantumVariationalKernel(QuantumVariationalKernel):
    def __init__(self, wires, ansatz, init_params, device, shots, logging_obj):
        self.device = qml.device(device, wires=wires, shots=shots)
        QuantumVariationalKernel.__init__(self, ansatz, init_params, self.device, logging_obj)

    def kernel_circuit(self, x1, x2, params):
        self.ansatz.ansatz(x1, params, self.wires)
        self.ansatz.adjoint_ansatz(x2, params, self.wires)
        return qml.probs(wires=self.wires)

    def kernel_with_params(self, x1, x2, params):
        return self.qnode(x1, x2, params)[0]

class SwapQuantumVariationalKernel(QuantumVariationalKernel):
    def __init__(self, wires, ansatz, init_params, device, shots, noise_model, logging_obj):
        self.noise = None
        if device == "qiskit.aer":
            self.device = qml.device(
                device,
                wires=2*wires+1,
                shots=shots,
                noise_model=build_noise_model_qiskit(noise_model, 2*wires+1),
                max_parallel_threads=10,
                max_parallel_experiments=1,
                max_parallel_shots=0
            )
        elif device == "default.mixed":
            self.noise = build_noise_model_qml(noise_model)
            ansatz.set_noise(self.noise)
            self.device = qml.device(device, wires=2*wires+1, shots=shots)
        else:
            self.device = qml.device(device, wires=2*wires+1, shots=shots)
        QuantumVariationalKernel.__init__(self, ansatz, init_params, self.device, logging_obj)
        self.first_wires = self.wires[1:1+wires]
        self.second_wires = self.wires[1+wires:]

    def kernel_circuit(self, x1, x2, params):
        self.ansatz.ansatz(x1, params, self.first_wires)
        self.ansatz.ansatz(x2, params, self.second_wires)

        NoiseExtension(self.noise).apply(wires=[0], gate=qml.Hadamard)
        for x, y in zip(self.first_wires, self.second_wires):
            NoiseExtension(self.noise).apply(wires=[0, x, y], gate=qml.CSWAP)
        NoiseExtension(self.noise).apply(wires=[0], gate=qml.Hadamard)

        return qml.probs(wires=[0])

    def kernel_with_params(self, x1, x2, params):
        return 1 - 2 * self.qnode(x1, x2, params)[1]