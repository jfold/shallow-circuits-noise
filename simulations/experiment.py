from .circuits import VQC
from .hamiltonian import Hamiltonian
from .imports import *
from .parameters import *


class Experiment(object):
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        self.p_noises_str = ["1e-8", "1e-7", "1e-4", "1e-3", "1e-2", "1e-1"]
        self.n_p_noises = len(self.p_noises_str)

        self.d = 2 ** (self.parameters.n_qubits)
        self.seeds = np.arange(self.parameters.n_seeds).astype(int)
        self.layers = np.arange(1, 1 + self.parameters.n_layers).astype(int)
        self.p_noises = np.array([float(x) for x in self.p_noises_str])
        self.spectras = np.full(
            (
                self.parameters.n_seeds,
                self.parameters.n_layers,
                self.n_p_noises,
                2 ** self.parameters.n_qubits,
            ),
            np.nan,
        )
        self.l1s = np.full(
            (self.parameters.n_seeds, self.parameters.n_layers, self.n_p_noises), np.nan
        )
        self.commutator_norms = np.full(
            (self.parameters.n_seeds, self.parameters.n_layers, self.n_p_noises), np.nan
        )
        self.fidelities = np.full(
            (self.parameters.n_seeds, self.parameters.n_layers, self.n_p_noises), np.nan
        )
        self.circ_err_rates = np.full(
            (self.parameters.n_layers, self.n_p_noises), np.nan
        )
        self.n_gates = np.full((self.parameters.n_layers), np.nan)

        self.summary = {
            "n_gates": self.n_gates,
            "circ_err_rates": self.circ_err_rates,
            "l1s": self.l1s,
            "spectras": self.spectras,
            "layers": self.layers,
            "seeds": self.seeds,
            "commutator_norms": self.commutator_norms,
            "fidelities": self.fidelities,
            "p_noises_str": self.p_noises_str,
        }

        self.hamiltonian = Hamiltonian(parameters)

    def kl_divergence_uniform(self, spectrum: np.ndarray) -> float:
        spectrum /= np.sum(spectrum)
        uniform_prob = 1 / (spectrum.size - 1)
        kl = -np.nansum(uniform_prob * np.log(spectrum / uniform_prob))
        return kl

    def lp_norm(self, spectrum: np.ndarray, p: float = 1.0) -> float:
        spectrum /= np.sum(spectrum)
        uniform_prob = 1 / spectrum.size
        dist = np.linalg.norm(spectrum - uniform_prob, ord=p)
        return dist / 2

    def bhattacharyya_distance(self, spectrum: np.ndarray) -> float:
        spectrum /= np.sum(spectrum)
        uniform_prob = 1 / spectrum.size
        dist = np.sum(np.sqrt(spectrum) * np.sqrt(uniform_prob))
        return dist

    def save(self):
        for k, v in self.summary.items():
            if isinstance(v, np.ndarray):
                self.summary.update({k: v.tolist()})
        json_dump = json.dumps(self.summary)
        with open(self.parameters.savepth + "results.json", "w") as f:
            f.write(json_dump)

    def load_LiH(self) -> Tuple[list, list]:
        data = np.loadtxt("chemistry_hamiltonians/LiH.txt")
        coeffs = data[:, 0]
        operators = data[:, 1:]
        num2op = {0: "I", 1: "X", 2: "Y", 3: "Z"}
        paulis = []
        for operator in operators:
            paulis.append("".join([num2op[op] for op in operator]))
        return coeffs, paulis

    def compute_commutator_norm(
        self, psi: np.ndarray, rho: np.ndarray, p: float = 1.0
    ) -> float:
        psi = psi[:, np.newaxis] if psi.ndim == 1 else psi
        rho_psi = rho @ psi
        F = (psi.T.conj() @ rho_psi).squeeze().real
        commutator_norm = (
            2 ** (1 / p) * np.sqrt((psi.T.conj() @ rho @ rho_psi).real - F ** 2)
        ).squeeze()
        return commutator_norm, F

    def run(self, save: bool = False):
        if self.parameters.hamiltonian == "LiH-":
            coeffs, paulis = self.load_LiH()
            hamiltonian = Hamiltonian(self.parameters, specifications=(coeffs, paulis))

        for i_p, p_noise in enumerate(self.p_noises):
            self.parameters.p_noise = p_noise
            for i_s, seed in enumerate(self.seeds):
                self.parameters.seed = seed

                vqc = (
                    VQC(self.parameters, hamiltonian=hamiltonian)
                    if self.parameters.hamiltonian == "LiH-"
                    else VQC(self.parameters)
                )
                layers_list = vqc.build_ansatz(return_layers=True)
                reg = Register(
                    num_qubits=self.parameters.n_qubits,
                    density_matrix=self.parameters.density_matrix,
                )
                # Pure version
                params = deepcopy(self.parameters)
                params.density_matrix = False
                vqc_pure = (
                    VQC(params, hamiltonian=hamiltonian)
                    if self.parameters.hamiltonian == "LiH-"
                    else VQC(params)
                )
                layers_list_pure = vqc_pure.build_ansatz(return_layers=True)
                reg_pure = Register(
                    num_qubits=self.parameters.n_qubits, density_matrix=False,
                )
                # Circuit
                for i_l, layer in enumerate(self.layers):
                    if layer == 1:
                        reg.apply_circuit(layers_list[0])
                    reg.apply_circuit(layers_list[layer])
                    rho = reg[:, :]

                    if layer == 1:
                        reg_pure.apply_circuit(layers_list_pure[0])
                    reg_pure.apply_circuit(layers_list_pure[layer])
                    psi = reg_pure[:]

                    (
                        self.commutator_norms[i_s, i_l, i_p],
                        self.fidelities[i_s, i_l, i_p],
                    ) = self.compute_commutator_norm(psi=psi, rho=rho)
                    self.spectras[i_s, i_l, i_p, :] = vqc.rho_spectrum(rho=rho)
                    self.l1s[i_s, i_l, i_p] = self.lp_norm(
                        self.spectras[i_s, i_l, i_p, 1:], p=1.0
                    )
                    self.n_gates[i_l] = vqc.n_gates_pr_layer * layer
                    self.circ_err_rates[i_l, i_p] = vqc.circuit_err_rate()

        if save:
            self.save()


def run_spinchain_experiments(qubits: np.ndarray = np.arange(3, 11)):
    n_e = 1
    for n_qubits in tqdm(qubits):
        for scramble in [False, True]:
            for hamiltonian in ["TFI-u", "XXZ-u", "TFI-uni", "XXZ-uni"]:
                mixer = (
                    "X" if hamiltonian == "TFI-u" or hamiltonian == "TFI-uni" else "Z"
                )
                for vqe in [False, True]:
                    for ansatz in ["SEL", "HVA"]:
                        if (ansatz == "SEL" and vqe) or (scramble and vqe):
                            continue
                        n_layers = 50 if ansatz == "SEL" else 100
                        settings = f"{ansatz}-{hamiltonian}-{n_e}"
                        parameters = Parameters(
                            {
                                "experiment": settings,
                                "ansatz": ansatz,
                                "vqe": vqe,
                                "hamiltonian": hamiltonian,
                                "n_qubits": n_qubits,
                                "n_layers": n_layers,
                                "mixer": mixer,
                                "two_qubit_noise": False,
                                "add_scrambling": scramble,
                            },
                            mkdir=True,
                        )
                        experiment = Experiment(parameters)
                        experiment.run(save=True)
                        n_e += 1


def run_chemistry_experiments():
    for n_e, vqe in [False, True]:
        settings = f"HVA-LiH-{n_e+1}"
        parameters = Parameters(
            {
                "experiment": settings,
                "ansatz": "HVA",
                "vqe": vqe,
                "hamiltonian": "LiH-",
                "n_qubits": 6,
                "n_seeds": 10,
                "n_layers": 50,
            },
            mkdir=True,
        )
        experiment = Experiment(parameters)
        experiment.run(save=True)
        n_e += 1

