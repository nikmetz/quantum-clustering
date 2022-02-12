import pennylane as qml
from math import gcd

def get_ansatz(name, num_wires, num_layers, params):
    if name == "ansatz1":
        return Ansatz1(num_wires, num_layers)
    elif name == "ansatz2":
        return Ansatz2(num_wires, num_layers)
    elif name == "ansatz3":
        return Ansatz3(num_wires, num_layers)
    elif name == "ansatz4":
        return Ansatz4(num_wires, num_layers, **params)

class Ansatz():
    def __init__(self, num_wires, num_layers):
        self.num_wires = num_wires
        self.num_layers = num_layers

    def ansatz(self, data, params):
        pass

    def adjoint_ansatz(self, data, params):
        qml.adjoint(self.ansatz)(data, params)

class Ansatz1(Ansatz):
    def __init__(self, num_wires, num_layers):
        Ansatz.__init__(self, num_wires, num_layers)

    def ansatz(self, data, params):
        for j, layer_params in enumerate(params):
            self.layer(data, layer_params, self.num_wires, i0=j * self.num_wires)

    def layer(self, x, params, wires, i0=0, inc=1):
        """Building block of the embedding ansatz"""
        i = i0
        for wire in range(wires):
            qml.Hadamard(wires=wire)
            qml.RZ(x[i % len(x)], wires=wire)
            i += inc
            qml.RY(params[wire, 0], wires=wire)

        qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=range(wires), parameters=params[:, 1])

class Ansatz2(Ansatz):
    def __init__(self, num_wires, num_layers):
        Ansatz.__init__(self, num_wires, num_layers)

    def ansatz2(self, data, params):
        for j, layer_params in enumerate(params):
            for wire in range(self.num_wires):
                qml.RX(layer_params[wire][0], wires=wire)
                qml.RY(layer_params[wire][1], wires=wire)
            for wire in range(0, self.num_wires - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
            for wire in range(1, self.num_wires - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
        qml.templates.embeddings.AngleEmbedding(
            features=data, wires=range(self.num_wires), rotation="X"
        )

class Ansatz3(Ansatz):
    def __init__(self, num_wires, num_layers):
        Ansatz.__init__(self, num_wires, num_layers)

    def ansatz3(self, data, params):
        for j, layer_params in enumerate(params):
            for wire in range(self.num_wires):
                qml.RX(layer_params[wire][0], wires=wire)
                qml.RY(layer_params[wire][1], wires=wire)
            for wire in range(0, self.num_wires - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
            for wire in range(1, self.num_wires - 1, 2):
                qml.CZ(wires=[wire, wire + 1])
            qml.templates.embeddings.AngleEmbedding(
                features=data, wires=range(self.num_wires), rotation="X"
            )

class Ansatz4(Ansatz):
    def __init__(self, num_wires, num_layers, r, gate_types):
        Ansatz.__init__(self, num_wires, num_layers)
        self.r = r
        self.gate_types = gate_types

    def ansatz4(self, data, params):
        for layer in range(self.num_layers):
            for wire in range(self.num_wires):
                getattr(qml, self.gate_types[layer][0])(params[layer][wire][0], wires=[wire])
            for i in range(int(self.num_wires/gcd(self.num_wires, self.r[layer]))):
                target = (i * self.r[layer] - self.r[layer]) % self.num_wires
                control = i * self.r[layer] % self.num_wires
                getattr(qml, self.gate_types[layer][1])(params[layer][wire][1], wires=[control, target])

            qml.templates.embeddings.AngleEmbedding(
                features=data, wires=range(self.num_wires), rotation="X"
            )