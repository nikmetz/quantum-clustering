import pennylane as qml
from math import gcd
from qclustering.NoiseExtension import NoiseExtension

def get_ansatz(name, num_layers, params):
    if name == "ansatz1":
        return Ansatz1(num_layers)
    elif name == "ansatz1noise":
        return Ansatz1Noise(num_layers)
    elif name == "ansatz2":
        return Ansatz2(num_layers)
    elif name == "ansatz3":
        return Ansatz3(num_layers)
    #elif name == "ansatz4":
    #    return Ansatz4(num_layers, **params)

class Ansatz():
    def __init__(self, num_layers, noise=None):
        self.num_layers = num_layers
        self.noise = None

    def set_noise(self, noise):
        self.noise = noise

    def ansatz(self, data, params, wires):
        pass

    def adjoint_ansatz(self, data, params, wires):
        adj = qml.adjoint(self.ansatz)
        adj(data, params, wires)

class Ansatz1(Ansatz):
    def __init__(self, num_layers):
        Ansatz.__init__(self, num_layers)

    def ansatz(self, data, params, wires):
        for j, layer_params in enumerate(params):
            self.layer(data, layer_params, wires, i0=j * len(wires))

    def layer(self, x, params, wires, i0=0, inc=1):
        """Building block of the embedding ansatz"""
        i = i0
        for idx, wire in enumerate(wires):
            qml.Hadamard(wires=[wire])
            qml.RZ(x[i % len(x)], wires=[wire])
            i += inc
            qml.RY(params[idx, 0], wires=[wire])

        qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[:, 1])

class Ansatz1Noise(Ansatz):
    def __init__(self, num_layers):
        Ansatz.__init__(self, num_layers)

    def ansatz(self, data, params, wires):
        for j, layer_params in enumerate(params):
            self.layer(data, layer_params, wires, i0=j * len(wires))

    def layer(self, x, params, wires, i0=0, inc=1):
        """Building block of the embedding ansatz"""
        i = i0
        for idx, wire in enumerate(wires):
            NoiseExtension(self.noise).apply(wires=[wire], gate=qml.Hadamard)
            NoiseExtension(self.noise).apply_parameterized(x[i % len(x)], wires=[wire], gate=qml.RZ)
            i += inc
            NoiseExtension(self.noise).apply_parameterized(params[idx, 0], wires=[wire], gate=qml.RY)

        ne = NoiseExtension(self.noise, qml.CRZ)
        qml.broadcast(unitary=ne.apply_parameterized, pattern="ring", wires=wires, parameters=params[:, 1])

class Ansatz2(Ansatz):
    def __init__(self, num_layers):
        Ansatz.__init__(self, num_layers)

    def ansatz(self, data, params, wires):
        for layer_params in params:
            for idx, wire in enumerate(wires):
                qml.RX(layer_params[idx][0], wires=[wire])
                qml.RY(layer_params[idx][1], wires=[wire])
            for wire in [(wires[i], wires[i+1]) for i in range(0, len(wires) - 1, 2)]:
                qml.CZ(wires=[wire[0], wire[1]])
            for wire in [(wires[i], wires[i+1]) for i in range(1, len(wires) - 1, 2)]:
                qml.CZ(wires=[wire[0], wire[1]])
        qml.templates.embeddings.AngleEmbedding(
            features=data, wires=wires, rotation="X"
        )

class Ansatz3(Ansatz):
    def __init__(self, num_layers):
        Ansatz.__init__(self, num_layers)

    def ansatz(self, data, params, wires):
        for layer_params in params:
            for idx, wire in enumerate(wires):
                qml.RX(layer_params[idx][0], wires=wire)
                qml.RY(layer_params[idx][1], wires=wire)
            for wire in [(wires[i], wires[i+1]) for i in range(0, len(wires) - 1, 2)]:
                qml.CZ(wires=[wire[0], wire[1]])
            for wire in [(wires[i], wires[i+1]) for i in range(1, len(wires) - 1, 2)]:
                qml.CZ(wires=[wire[0], wire[1]])
            qml.templates.embeddings.AngleEmbedding(
                features=data, wires=wires, rotation="X"
            )

class Ansatz4(Ansatz):
    def __init__(self, num_layers, r, gate_types):
        Ansatz.__init__(self, num_layers)
        self.r = r
        self.gate_types = gate_types

    def ansatz(self, data, params, wires):
        for layer in range(self.num_layers):
            for wire in wires:
                getattr(qml, self.gate_types[layer][0])(params[layer][wire][0], wires=[wire])
            for i in range(int(len(wires)/gcd(len(wires), self.r[layer]))):
                target = wires[(i * self.r[layer] - self.r[layer]) % len(wires)]
                control = wires[i * self.r[layer] % len(wires)]
                getattr(qml, self.gate_types[layer][1])(params[layer][wire][1], wires=[control, target])

            qml.templates.embeddings.AngleEmbedding(
                features=data, wires=wires, rotation="X"
            )