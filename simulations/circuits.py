from .hamiltonian import Hamiltonian
from .imports import *
from .optimizer import Optimizer
from .parameters import Parameters


class VQC(Optimizer):
    def __init__(
        self,
        parameters: Parameters,
        thetas: Dict = None,
        hamiltonian: Hamiltonian = None,
    ):
        Optimizer.__init__(self, parameters)
        self.seed = parameters.seed
        self.p_noise = parameters.p_noise
        self.noise = parameters.noise
        self.n_layers = parameters.n_layers
        self.n_qubits = parameters.n_qubits
        self.density_matrix = parameters.density_matrix
        self.ansatz = parameters.ansatz.lower()
        self.add_scrambling = parameters.add_scrambling
        self.two_qubit_noise = parameters.two_qubit_noise
        self.noise_channel = getattr(decoherence, self.noise)
        self.vqe = parameters.vqe
        self.dt = parameters.dt  # 1 / np.sqrt(self.n_layers)
        self.mixer = parameters.mixer
        self.hamiltonian = (
            hamiltonian if hamiltonian is not None else Hamiltonian(parameters)
        )
        self.hamiltonian_name = (
            hamiltonian.name.lower()
            if hamiltonian is not None
            else parameters.hamiltonian.lower()
        )
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.init_thetas(thetas)

    def set_n_gates(self):
        if self.ansatz.lower() == "hva":
            self.n_gates_pr_layer = self.hamiltonian.n_terms
            self.n_gates = self.n_layers * self.n_gates_pr_layer
            if self.hamiltonian_name in ["LiH-"]:  # chemistry
                self.n_gates_pr_layer = 0
                for pauli in self.hamiltonian.pauli_dict.keys():
                    n = int(
                        len(pauli.replace("(", "-").replace(")", "-").split("-")) / 3
                    )
                    self.n_gates_pr_layer += n * 4 + 1
                self.n_gates = self.n_layers * self.n_gates_pr_layer
        elif self.ansatz.lower() == "sel":
            self.n_gates_pr_layer = 4 * self.n_qubits
            self.n_gates = self.n_layers * self.n_gates_pr_layer
        elif self.ansatz.lower() == "test":
            self.n_gates_pr_layer = 1
            self.n_gates = 1
        else:
            raise ValueError(f"Ansatz {self.ansatz} not supported")

    def __str__(self):
        return str(self.build_ansatz())

    def get_operator(
        self,
        gate: str,
        q1: int,
        q2: int = None,
        theta: float = None,
        dU_di: bool = False,
    ) -> list:
        operator = []

        assert q1 <= self.n_qubits and gate in [
            "Rzz",
            "Ryy",
            "Rxx",
            "Rz",
            "Ry",
            "Rx",
            "CNOT",
        ]
        next_q = int((q1 + 1) % self.n_qubits) if q2 is None else q2

        if gate == "Rzz":
            if dU_di:
                operator.extend(
                    [Z(q1), Z(next_q)]
                )  # not sure this is correct but not used
            operator.append(Rzz(q1, next_q, theta))
        elif gate == "Ryy":
            if dU_di:
                operator.extend(
                    [Y(q1), Y(next_q)]
                )  # not sure this is correct but not used
            operator.append(Ryy(q1, next_q, theta))
        elif gate == "Rxx":
            if dU_di:
                operator.extend(
                    [X(q1), X(next_q)]
                )  # not sure this is correct but not used
            operator.append(Rxx(q1, next_q, theta))
        elif gate == "Rx":
            if dU_di:
                operator.append(X(q1))
            operator.append(Rx(q1, theta))
        elif gate == "Ry":
            if dU_di:
                operator.append(Y(q1))
            operator.append(Ry(q1, theta))
        elif gate == "Rz":
            if dU_di:
                operator.append(Z(q1))
            operator.append(Rz(q1, theta))
        elif gate == "CNOT":
            operator.append(X(target=q1, controls=next_q))
        else:
            raise ValueError(f"Gate {gate} not supported.")

        #### ADD NOISE CHANNEL
        if self.density_matrix:
            if self.two_qubit_noise:
                operator.append(self.noise_channel([q1, next_q], self.p_noise))
            else:
                operator.append(self.noise_channel(q1, self.p_noise))
                if len(gate) > 2:
                    operator.append(self.noise_channel(next_q, self.p_noise))

        return operator

    def init_thetas(self, thetas: Dict = None):
        self.set_n_gates()
        self.init_symbols()
        # Default
        upper_lim = 0.04 if self.vqe else 2 * np.pi
        self.thetas = {
            x: np.random.uniform(low=1e-4, high=upper_lim,) for x in self.symbols
        }
        # Special cases:
        if thetas is not None:
            assert thetas.keys() == self.thetas.keys()
            self.thetas = thetas
        elif self.ansatz == "HVA" and self.vqe:
            for l in range(self.n_layers):
                for pauli, coeff in self.hamiltonian.pauli_dict.items():
                    key = "R" + f"{pauli}[{l}]".lower()
                    prefac = (
                        self.dt
                        if pauli.split("(")[0] == self.mixer
                        else (self.dt * (l + 1)) / self.n_layers
                    )
                    self.thetas.update({key: prefac * coeff})

    def init_symbols(self):
        self.symbols = []
        for l in range(self.n_layers):
            for q in range(self.n_qubits):
                next_q = int((q + 1) % self.n_qubits)
                if self.ansatz == "hva":
                    if "tfi-" in self.hamiltonian_name:
                        self.symbols.append(f"Rzz({q},{next_q})[{l}]")
                        self.symbols.append(f"Rx({q})[{l}]")
                        if self.add_scrambling:
                            self.symbols.append(f"Rz({q})[{l}]")
                    elif "xxz-" in self.hamiltonian_name:
                        self.symbols.append(f"Rzz({q},{next_q})[{l}]")
                        self.symbols.append(f"Ryy({q},{next_q})[{l}]")
                        self.symbols.append(f"Rxx({q},{next_q})[{l}]")
                        if self.add_scrambling:
                            self.symbols.append(f"Rx({q})[{l}]")
                        self.symbols.append(f"Rz({q})[{l}]")
                elif self.ansatz.lower() == "sel":
                    self.symbols.append(f"Ry({q})[{l},1]")
                    self.symbols.append(f"Rz({q})[{l}]")
                    self.symbols.append(f"Ry({q})[{l},2]")
                elif self.ansatz == "test":
                    self.symbols.append(f"theta({q})[{l}]")

    def add_gates(self, gate: str, thetas: Dict, name: str) -> list:
        circ_list = []
        for q in range(self.n_qubits):
            next_q = int((q + 1) % self.n_qubits)
            if "R" in gate:
                cur_theta = (
                    f"{gate}({q},{next_q})[{name}]"
                    if len(gate) == 3
                    else f"{gate}({q})[{name}]"
                )
                if cur_theta not in thetas.keys():
                    print(self.__dict__)
                    print(thetas.keys())
                circ_list.extend(self.get_operator(gate, q1=q, theta=thetas[cur_theta]))
            else:
                circ_list.extend(self.get_operator(gate, q1=q))

        return circ_list

    def init_mixer(self):
        self.mix_init_coeff = self.hamiltonian.H_dict[self.mixer]
        if self.mixer == "X":
            circ_list = []
            for i in range(self.n_qubits):
                ang = -np.pi / 2 if self.mix_init_coeff[i] > 0 else np.pi / 2
                circ_list.append(Ry(i, ang))
        elif self.mixer == "Z":
            circ_list = []
            for i in range(self.n_qubits):
                ang = np.pi if self.mix_init_coeff[i] > 0 else 0
                circ_list.append(Ry(i, ang))
        return circ_list

    def HVA_TFI(self, thetas: Dict = None, dU_di: int = -1) -> list:
        circ_list = self.init_mixer()
        layers_list = [Circuit(circ_list)]

        if thetas is None:
            thetas = self.thetas

        for l in range(self.n_layers):
            layer = [
                *self.add_gates(gate="Rzz", thetas=thetas, name=l),
                *self.add_gates(gate="Rx", thetas=thetas, name=l),
            ]
            if self.add_scrambling:
                layer.extend(self.add_gates(gate="Rz", thetas=thetas, name=l))
            circ_list.extend(layer)
            layers_list.append(Circuit(layer))

        return circ_list, layers_list

    def HVA_XXZ(self, thetas: Dict = None, dU_di: int = -1) -> list:
        circ_list = self.init_mixer()
        layers_list = [Circuit(circ_list)]

        if thetas is None:
            thetas = self.thetas

        for l in range(self.n_layers):
            layer = [
                *self.add_gates(gate="Rzz", thetas=thetas, name=l),
                *self.add_gates(gate="Ryy", thetas=thetas, name=l),
                *self.add_gates(gate="Rxx", thetas=thetas, name=l),
                *self.add_gates(gate="Rz", thetas=thetas, name=l),
            ]
            if self.add_scrambling:
                layer.extend(self.add_gates(gate="Rx", thetas=thetas, name=l))
            circ_list.extend(layer)
            layers_list.append(Circuit(layer))

        return circ_list, layers_list

    def SEL(self, thetas: Dict = None, dU_di: int = -1) -> Tuple[list, list]:
        # https://pennylane.readthedocs.io/en/stable/code/api/pennylane.StronglyEntanglingLayers.html
        circ_list = []
        layers_list = [Circuit(circ_list)]

        if thetas is None:
            thetas = self.thetas
        for l in range(self.n_layers):
            layer = [
                *self.add_gates(gate="Ry", thetas=thetas, name=f"{l},1"),
                *self.add_gates(gate="Rz", thetas=thetas, name=l),
                *self.add_gates(gate="Ry", thetas=thetas, name=f"{l},2"),
                *self.add_gates(gate="CNOT", thetas=thetas, name=l),
            ]
            circ_list.extend(layer)
            layers_list.append(Circuit(layer))

        return circ_list, layers_list

    def HVA_LiH(self, thetas: Dict = None, dU_di: int = -1) -> Tuple[list, list]:
        if thetas is None:
            thetas = self.thetas
        circ_list = []
        layers_list = [Circuit(circ_list)]
        self.n_gates = 0
        coeffs = np.abs(self.hamiltonian.coefficients[1:])
        paulis = self.hamiltonian.operators[1:]
        probs = coeffs / np.sum(coeffs)
        n_pauli_terms_pr_layer = 100

        for l in range(self.n_layers):
            samples = np.random.choice(
                range(coeffs.size),
                size=(n_pauli_terms_pr_layer,),
                p=probs,
                replace=False,
            )
            layer = []
            for ops_idx in samples:
                paulis_ = paulis[ops_idx]
                coeffs_ = coeffs[ops_idx]
                prod = None
                qs = []
                for q, s in enumerate(paulis_):
                    if s != "I":
                        op = getattr(unitaries, s)
                        qs.append(q)
                        if prod is None:
                            prod = op(q)
                        else:
                            prod = prod * op(q)

                if prod is not None:
                    if not self.vqe:
                        theta = np.random.uniform(low=1e-4, high=2 * np.pi)
                        layer.append(R(prod, theta))
                    else:
                        layer.append(
                            R(prod, coeffs_ * (self.dt * (l + 1)) / self.n_layers)
                        )
                    self.n_gates += 1

                if self.density_matrix:
                    for q in qs:
                        layer.append(self.noise_channel(q, self.p_noise))

            circ_list.extend(layer)
            layers_list.append(Circuit(layer))

        if False:  # old
            for l in range(self.n_layers):
                layer = []
                for pauli, coeff in zip(ops, coeffs):
                    prod = None
                    qs = []
                    for q, s in enumerate(pauli):
                        if s != "I":
                            op = getattr(unitaries, s)
                            qs.append(q)
                            if prod is None:
                                prod = op(q)
                            else:
                                prod = prod * op(q)

                    if prod is not None:
                        if not self.vqe:
                            theta = np.random.uniform(low=1e-4, high=2 * np.pi)
                            layer.append(R(prod, theta))
                        else:
                            layer.append(
                                R(prod, coeff * (self.dt * (l + 1)) / self.n_layers)
                            )
                        self.n_gates += 1

                    for q in qs:
                        layer.append(self.noise_channel(q, self.p_noise))

                circ_list.extend(layer)
                layers_list.append(Circuit(layer))

        return circ_list, layers_list

    def build_ansatz(
        self, thetas: Dict = None, dU_di: int = -1, return_layers: bool = False
    ) -> list:
        if "sel" in self.ansatz:
            ansatz, layers_list = self.SEL(thetas, dU_di)
        elif "hva" in self.ansatz:
            if "tfi-" in self.hamiltonian_name:
                ansatz, layers_list = self.HVA_TFI(thetas, dU_di)
            elif "xxz-" in self.hamiltonian_name:
                ansatz, layers_list = self.HVA_XXZ(thetas, dU_di)
            elif "lih-" in self.hamiltonian_name:
                ansatz, layers_list = self.HVA_LiH(thetas, dU_di)
        else:
            raise ValueError(f"Ansatz {self.ansatz} not supported")
        if return_layers:
            return layers_list
        return ansatz

    def build_circuit(self, ansatz: list = None, thetas: Dict = None):
        if ansatz is None:
            circ = Circuit(self.build_ansatz(thetas=thetas))
        else:
            circ = Circuit(ansatz)
        return circ

    def rho_spectrum(self, rho: np.ndarray = None, ansatz: list = None):
        if rho is None:
            rho = self.run_circuit(ansatz)
        spectrum = np.linalg.eigvalsh(rho)
        return np.sort(spectrum)[::-1]

    def shannon_entropy(self, sub_idx: int = 1, threshold: float = 1e-50):
        spectrum = self.rho_spectrum()
        subspectrum = spectrum[sub_idx:]
        subspectrum = subspectrum[subspectrum > threshold]
        subspectrum /= np.sum(subspectrum)
        if subspectrum.size == 0:
            entropy = 0.0
        else:
            entropy = -np.inner(subspectrum, np.log10(subspectrum))
        return entropy

    def renyi_entropy(
        self, sub_idx: int = 1, threshold: float = 1e-50, alpha: float = 2.0
    ):
        spectrum = self.rho_spectrum()
        subspectrum = spectrum[sub_idx:]
        subspectrum = subspectrum[subspectrum > threshold]
        subspectrum /= np.sum(subspectrum)

        if subspectrum.size == 0:
            entropy = 0.0
        else:
            entropy = (1 / (1 - alpha)) * np.sum(np.log10(subspectrum ** alpha))
        return entropy

    def circuit_err_rate(self):
        return self.p_noise * self.n_gates

    def run_circuit(self, ansatz: list = None) -> np.ndarray:
        reg = Register(num_qubits=self.n_qubits, density_matrix=self.density_matrix)
        circ = self.build_circuit(ansatz)
        reg.apply_circuit(circ)
        if self.density_matrix:
            rho = reg[:, :]
            return rho
        else:
            psi = reg[:]
            return psi


# class PVQC(Optimizer):
#     def __init__(self, parameters: Parameters):
#         Optimizer.__init__(self, parameters)
#         self.seed = parameters.seed
#         self.p_noise = parameters.p_noise
#         self.noise = parameters.noise
#         self.n_layers = parameters.n_layers
#         self.n_qubits = parameters.n_qubits
#         self.density_matrix = parameters.density_matrix

#         np.random.seed(self.seed)
#         random.seed(self.seed)
#         self.thetas = np.random.uniform(
#             low=1e-4, high=5e-2, size=(3 * self.n_layers * self.n_qubits)
#         )
#         self.nus = np.random.uniform(
#             low=1e-10, high=1 - 1e-10, size=(self.n_layers * self.n_qubits)
#         )

#     def sample_ansatz(
#         self, thetas: Dict = None, nus: np.ndarray = None, dU_di: int = -1
#     ) -> list:
#         noise_channel = getattr(decoherence, self.noise)
#         circ_list = [H(i) for i in range(self.n_qubits)]
#         if thetas is None:
#             thetas = self.thetas
#         if nus is None:
#             nus = self.nus

#         betas = np.random.uniform(low=0.0, high=1.0, size=(self.nus.size,)) < self.nus
#         i_t = 0
#         i_n = 0
#         for l in range(self.n_layers):
#             for q in range(self.n_qubits):
#                 if i_t == dU_di:
#                     circ_list.append(Z(q))
#                 circ_list.append(Rz(q, thetas[i_t]))
#                 if self.density_matrix:
#                     circ_list.append(noise_channel(q, self.p_noise))
#                 i_t += 1

#                 if i_t == dU_di:
#                     circ_list.append(Y(q))
#                 circ_list.append(Ry(q, thetas[i_t]))
#                 if self.density_matrix:
#                     circ_list.append(noise_channel(q, self.p_noise))
#                 i_t += 1

#                 if i_t == dU_di:
#                     circ_list.append(Z(q))
#                 circ_list.append(Rz(q, thetas[i_t]))
#                 if self.density_matrix:
#                     circ_list.append(noise_channel(q, self.p_noise))
#                 i_t += 1

#                 if q < self.n_qubits - 1:
#                     circ_list.extend(
#                         [X(q, q + 1)]  # , Rz(q, thetas[i_t]), Rz(q, thetas[i_t])]
#                     )
#                     if self.density_matrix:
#                         circ_list.append(noise_channel(q, self.p_noise))
#                 elif self.n_qubits > 2:
#                     circ_list.extend(
#                         [X(q, 0)]  # ,Rz(q, thetas[i_t]),  Rz(q, thetas[i_t])]
#                     )
#                     if self.density_matrix:
#                         circ_list.append(noise_channel(q, self.p_noise))
#                 # i_t += 1

#                 if betas[i_n]:
#                     circ_list.append(M(q))
#                 i_n += 1
#         return circ_list

#     def build_circuit(self, thetas: Dict = None):
#         circ = Circuit(self.sample_ansatz(thetas))
#         return circ

#     def run_circuit(self) -> np.ndarray:
#         reg = Register(num_qubits=self.n_qubits, density_matrix=self.density_matrix)
#         circ = self.build_circuit()
#         reg.apply_circuit(circ)
#         if self.density_matrix:
#             rho = reg[:, :]
#             return rho
#         else:
#             psi = reg[:]
#             return psi
