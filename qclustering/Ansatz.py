import pennylane as qml

def get_ansatz(name):
    if name == "ansatz1":
        return ansatz
    elif name == "ansatz2":
        return ansatz2
    elif name == "ansatz3":
        return ansatz3

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for wire in range(wires):
        qml.Hadamard(wires=wire)
        qml.RX(x[i % len(x)], wires=wire)
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