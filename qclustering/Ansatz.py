import pennylane as qml
from math import gcd

def get_ansatz(name, num_wires, num_layers, params):
    if name == "ansatz1":
        return lambda x, p: ansatz(x, p, num_wires)
    elif name == "ansatz2":
        return lambda x, p: ansatz2(x, p, num_wires)
    elif name == "ansatz3":
        return lambda x, p: ansatz3(x, p, num_wires)
    elif name == "ansatz4":
        return lambda x, p: ansatz4(x, p, num_wires, num_layers, **params)

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for wire in range(wires):
        qml.Hadamard(wires=wire)
        qml.RZ(x[i % len(x)], wires=wire)
        i += inc
        qml.RY(params[wire, 0], wires=wire)

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=range(wires), parameters=params[:, 1])

def ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * wires)

def ansatz2(data, params, wires):
    for j, layer_params in enumerate(params):
        for wire in range(wires):
            qml.RX(layer_params[wire][0], wires=wire)
            qml.RY(layer_params[wire][1], wires=wire)
        for wire in range(0, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
    qml.templates.embeddings.AngleEmbedding(
        features=data, wires=range(wires), rotation="X"
    )

def ansatz3(data, params, wires):
    for j, layer_params in enumerate(params):
        for wire in range(wires):
            qml.RX(layer_params[wire][0], wires=wire)
            qml.RY(layer_params[wire][1], wires=wire)
        for wire in range(0, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        for wire in range(1, wires - 1, 2):
            qml.CZ(wires=[wire, wire + 1])
        qml.templates.embeddings.AngleEmbedding(
            features=data, wires=range(wires), rotation="X"
        )

def ansatz4(data, params, num_wires, num_layers, r, gate_types):
    for layer in range(num_layers):
        for wire in range(num_wires):
            getattr(qml, gate_types[layer][0])(params[layer][wire][0], wires=[wire])
        for i in range(int(num_wires/gcd(num_wires, r[layer]))):
            target = (i * r[layer] - r[layer]) % num_wires
            control = i * r[layer] % num_wires
            getattr(qml, gate_types[layer][1])(params[layer][wire][1], wires=[control, target])

        qml.templates.embeddings.AngleEmbedding(
            features=data, wires=range(num_wires), rotation="X"
        )