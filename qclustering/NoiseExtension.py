class NoiseExtension():
    def __init__(self, noise_model, gate=None):
        self.noise_model = noise_model
        self.gate = gate

    def apply_error(self, gate_basis, wires):
        if self.noise_model is not None:
            error_list = []
            error_list += self.noise_model.incoherent
            if gate_basis is not None:
                error_list += self.noise_model.coherent[gate_basis]

            for error in error_list:
                for wire in wires:
                    error(wires=[wire])

    def apply_parameterized(self, params, wires, gate=None):
        if gate is None:
            gate = self.gate

        self.apply_error(gate.basis, wires)

        gate(params, wires=wires)

    def apply(self, wires, gate=None):
        if gate is None:
            gate = self.gate

        self.apply_error(gate.basis, wires)

        gate(wires=wires)