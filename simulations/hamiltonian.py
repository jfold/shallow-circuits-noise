from .parameters import *


class Hamiltonian(object):
    """# Hamiltonian for a spin ring """

    # https://questlink.qtechtheory.org/demo_hamiltonian.txt

    def __init__(
        self, parameters: Parameters, specifications: Tuple[list, list] = (None, None),
    ):
        self.__dict__.update(asdict(parameters))
        self.qubits = cirq.GridQubit.rect(self.n_qubits, 1)
        self.H_dict = {}
        self.H = cirq.PauliSum()
        self.init_H_dict()
        self.build_H(coefficients=specifications[0], operators=specifications[1])
        self.name = parameters.hamiltonian

    def __str__(self) -> str:
        if hasattr(self, "H"):
            return str(self.H)

    def init_H_dict(self):
        self.H_type, self.coeff_type = self.hamiltonian.split("-")

        # quantum chemistry example

        # spin chain
        if self.H_type == "X":
            self.H_dict.update({"X": None})
        elif self.H_type == "Z":
            self.H_dict.update({"Z": None})
        elif self.H_type == "IC":
            self.H_dict.update({"Z": None, "ZZ": None})
        elif self.H_type == "TFI":
            self.H_dict.update({"ZZ": None, "X": -np.ones(self.n_qubits)})
        # elif self.H_type == "Heisenberg":
        #     self.H_dict.update({"Z": None, "ZZ": None, "XX": None, "YY": None})
        elif self.H_type == "XXZ":
            self.H_dict.update(
                {
                    "Z": None,
                    "ZZ": -np.ones(self.n_qubits),
                    "XX": -np.ones(self.n_qubits),
                    "YY": -np.ones(self.n_qubits),
                }
            )
        elif self.H_type == "LiH":
            pass
        else:
            raise ValueError(f"Hamiltonian {self.H_type} not supported.")

    def build_H_dict(self, loc: float = 0, scale: float = 1):
        np.random.seed(self.seed)
        for key in self.H_dict.keys():
            if self.H_dict[key] is None:
                if self.coeff_type == "u":
                    coeffs = -np.ones(self.n_qubits)
                elif self.coeff_type == "uni":
                    coeffs = -np.random.uniform(low=-1.0, high=1.0, size=self.n_qubits)
                elif self.coeff_type == "nor":
                    coeffs = -np.random.normal(loc=loc, scale=scale, size=self.n_qubits)
                if self.H_type == "XXZ":
                    coeffs = np.tile(coeffs[0], self.n_qubits)
                self.H_dict.update({key: coeffs})  # / np.linalg.norm(coeffs)

    def build_H(
        self, operators: list = None, coefficients: list = None
    ) -> cirq.PauliSum:
        self.pauli_dict = {}
        if (
            operators is None
            or coefficients is None
            or len(operators) != len(coefficients)
        ):
            self.build_H_dict()
            for pauli, coeffs in self.H_dict.items():
                for i in range(self.n_qubits):
                    if len(pauli) == 1:
                        obs = getattr(cirq, pauli)
                        observable = float(coeffs[i]) * obs(self.qubits[i])
                        self.pauli_dict.update({f"{pauli}({i})": coeffs[i]})
                    if len(pauli) == 2:
                        obs_1 = getattr(cirq, pauli[0])
                        obs_2 = getattr(cirq, pauli[1])
                        if i < self.n_qubits - 1:
                            observable = (
                                float(coeffs[i])
                                * obs_1(self.qubits[i])
                                * obs_2(self.qubits[i + 1])
                            )
                            self.pauli_dict.update({f"{pauli}({i},{i+1})": coeffs[i]})
                        else:
                            observable = (
                                float(coeffs[i])
                                * obs_1(self.qubits[i])
                                * obs_2(self.qubits[0])
                            )
                            self.pauli_dict.update({f"{pauli}({i},{0})": coeffs[i]})
                    self.H += observable

        else:
            self.operators = operators
            self.coefficients = coefficients

            for coeff, operator in zip(coefficients, operators):
                observables = [float(coeff)]
                pauli_strs = []
                for q, op in enumerate(operator):
                    if op != "I":
                        obs = getattr(cirq, op)
                        observables.append(obs(self.qubits[q]))
                        pauli_strs.append(f"{obs}({q})")

                self.H += np.prod(observables)
                self.pauli_dict.update({"".join(pauli_strs): float(coeff)})

        self.n_terms = len(self.pauli_dict)
        self.H_mat = self.H.matrix()
        spectrum, eigenstates = np.linalg.eigh(self.H_mat)
        sort_args = np.argsort(spectrum)
        self.spectrum = spectrum[sort_args]
        self.eigenstates = eigenstates[:, sort_args]
        self.groundstate = self.eigenstates[:, [0]]

    def expectation(self, state: np.ndarray) -> float:
        if state.ndim == 2:
            return np.trace(state @ self.H_mat).real
        else:
            state = state.reshape(-1, 1)
            state_T = state.transpose().conjugate()
            return np.trace(state_T @ self.H_mat @ state).real

    # def cirq_expectation(
    #     self,
    #     unitary: cirq.Circuit,
    #     theta_symbols: list,
    #     theta_values: np.ndarray,
    #     p_err: float,
    # ):
    #     parameter_strs = tf.Variable(theta_values)
    #     parameter_vals = tf.constant(theta_symbols)
    #     expectation = tfq.noise.Expectation(
    #         differentiator=tfq.differentiators.Adjoint(),
    #         backend=DensityMatrixSimulator(noise=cirq.depolarize(p_err)),
    #     )(
    #         unitary,
    #         operators=self.H,
    #         symbol_names=parameter_strs,
    #         symbol_values=parameter_vals,
    #     )
    #     return expectation
