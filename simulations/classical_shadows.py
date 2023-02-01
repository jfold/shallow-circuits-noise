from .parameters import Parameters
from .hamiltonian import Hamiltonian
from .imports import *


class ClassicalShadows(object):
    def __init__(self, parameters: Parameters) -> None:
        self.shadow_size = int(parameters.shadow_size)
        self.n_chunks = parameters.n_chunks
        assert self.shadow_size % self.n_chunks == 0
        self.chunk_size = int(self.shadow_size / self.n_chunks)
        self.density_matrix = parameters.density_matrix
        self.n_qubits = parameters.n_qubits
        self.qubits = cirq.GridQubit.rect(self.n_qubits, 1)
        self.seed = parameters.seed
        np.random.seed(self.seed)

        self.X = np.array([[0, -1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.local_observables = {"X": 0, "Y": 1, "Z": 2}
        self.hermitians = [self.X, self.Y, self.Z]
        self.I = np.identity(2, dtype=complex)

        self.sgate = np.array([[1, 0], [0, -1j]], dtype=complex)
        self.hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

        self.zero_state = np.array([[1, 0], [0, 0]], dtype=complex)
        self.one_state = np.array([[0, 0], [0, 1]], dtype=complex)
        self.unitaries = [self.hadamard, self.hadamard @ self.sgate, self.I]

    def generate_observables(self):
        return np.random.choice(
            list(self.local_observables.keys()), size=(self.shadow_size, self.n_qubits)
        )

    def add_measurement(self, ansatz: list, measurements: list) -> list:
        ansatz_ = ansatz.copy()
        for measurement, qubit in zip(measurements, range(self.n_qubits)):
            if measurement == "X":
                ansatz_.extend([H(qubit), M(qubit)])
            if measurement == "Y":
                ansatz_.extend([S(qubit), Z(qubit), H(qubit), M(qubit)])
            if measurement == "Z":
                ansatz_.extend([M(qubit)])
        return ansatz_

    def generate_measurements(self, ansatz: list) -> np.ndarray:
        observables = self.generate_observables()
        measurements = []
        for observable in observables:
            register = Register(self.n_qubits, density_matrix=self.density_matrix)
            ansatz_ = self.add_measurement(ansatz, observable)
            measurement = register.apply_circuit(Circuit(ansatz_))
            measurement = 1 + (-2) * np.array(measurement)
            measurements.append(measurement)
        return np.array(measurements).squeeze(), observables

    def get_snapshot(self, observable: str, measurement: np.ndarray):
        snapshot = [1]
        for q in range(self.n_qubits):
            rho_q = self.zero_state if measurement[q] == 1 else self.one_state
            U_j = self.unitaries[self.local_observables[observable[q]]]
            local_snapshot = 3 * (U_j.conj().T @ rho_q @ U_j) - self.I
            snapshot = np.kron(snapshot, local_snapshot)
        return snapshot

    def state_tomography(self, ansatz: list):
        measurements, observables = self.generate_measurements(ansatz)
        rho_hat = np.zeros((2 ** self.n_qubits, 2 ** self.n_qubits), dtype=complex)
        for observable, measurement in zip(observables, measurements):
            rho_hat += self.get_snapshot(observable, measurement) / self.shadow_size
        return rho_hat

    def get_pauli_expectation(
        self, paulis: list, unitary: list, measurements: list
    ) -> float:
        prod = 1
        for pauli in paulis:
            q = int(pauli[pauli.find("(") + 1 : pauli.rfind(")")])
            if unitary[q] != pauli[0]:
                return 0
            prod *= 3 * measurements[q]
            # basis = unitary[q]
            # rho_q = self.zero_state if measurements[q] == 1 else self.one_state
            # U_j = self.unitaries[self.local_observables[basis]]
            # P_j = self.hermitians[self.local_observables[basis]]
            # local_snapshot = 3*(P_j @ U_j.conj().T @ rho_q @ U_j)
            # prod *= np.trace(local_snapshot).real

        return prod

    def estimate_expectation(self, hamiltonian: Hamiltonian, ansatz: list) -> float:
        paulis = np.array([pauli for pauli in hamiltonian.pauli_dict.keys()]).flatten()
        coeffs = np.array(
            [coeffs for coeffs in hamiltonian.pauli_dict.values()]
        ).flatten()
        expecations = np.full((self.shadow_size, coeffs.size), np.nan)
        measurements, unitaries = self.generate_measurements(ansatz)
        for i_m, (unitary, measurement) in enumerate(zip(unitaries, measurements)):
            for i_p, local_pauli in enumerate(paulis):
                expecations[i_m, i_p] = self.get_pauli_expectation(
                    local_pauli.split("*"), unitary, measurement
                )

        expecations_means = []
        for i in range(0, self.shadow_size, self.chunk_size):
            expecations_means.append(
                np.mean(expecations[i : i + self.chunk_size, :], axis=0)
            )

        expecations_medians = np.median(np.array(expecations_means), axis=0)
        expectation = np.inner(coeffs, expecations_medians)
        return expectation
