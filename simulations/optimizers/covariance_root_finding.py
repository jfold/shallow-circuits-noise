from ..parameters import Parameters
from ..hamiltonian import Hamiltonian
from ..imports import *


class CovarianceRootFinding(object):
    def __init__(self, parameters: Parameters):
        self.n_qubits = parameters.n_qubits
        self.qubits = cirq.GridQubit.rect(self.n_qubits, 1)
        self.pool_size = parameters.pool_size
        self.seed = parameters.seed
        self.density_matrix = parameters.density_matrix
        self.pool = self.sample_pool()

    def compute_f_k(self, O_k: np.ndarray, H: np.ndarray, state: np.ndarray) -> float:
        if state.ndim == 2:
            f_k = np.trace(state @ O_k @ H) - (
                np.trace(state @ O_k) * np.trace(state @ H)
            )
        elif state.ndim == 1:
            psi_ = state.conjugate().transpose()
            f_k = psi_ @ O_k @ H @ state - (psi_ @ O_k @ state) * (psi_ @ H @ state)
        else:
            raise ValueError()
        return f_k

    def compute_f(self, hamiltonian: Hamiltonian, state: np.ndarray = None):
        f = np.full((self.N_c,), np.nan, dtype=np.complex_)
        if state is None:
            state = self.run_circuit()
        for k, O_k in enumerate(self.pool):
            circ = Circuit(O_k)
            O_k_mat = circ.as_matrix(self.n_qubits)
            f_k = self.compute_f_k(O_k_mat, hamiltonian.H_mat, state)
            f[k] = f_k
        f = np.append(np.real(f), np.imag(f), axis=0)
        return np.array(f)

    def compute_jacobian(self, hamiltonian: Hamiltonian) -> np.ndarray:
        diff_states = self.finite_difference()
        n_parameters = self.thetas.size
        J = np.full((2 * self.N_c, n_parameters), np.nan, dtype=np.complex_)
        for n in range(diff_states.shape[0]):
            J[:, n] = self.compute_f(hamiltonian, diff_states[n])

        return J

    def finite_difference(self) -> np.ndarray:
        n_parameters = self.thetas.size
        if self.density_matrix:
            diff_states = np.full(
                (n_parameters, 2 ** self.n_qubits, 2 ** self.n_qubits),
                np.nan,
                dtype=np.complex_,
            )
        else:
            diff_states = np.full(
                (n_parameters, 2 ** self.n_qubits), np.nan, dtype=np.complex_,
            )

        coeffs = self.fd_coeffs[0]
        for i in range(n_parameters):
            if self.density_matrix:
                rho_i = np.zeros(
                    (2 ** self.n_qubits, 2 ** self.n_qubits), dtype=np.complex128
                )
            else:
                rho_i = np.zeros((2 ** self.n_qubits,), dtype=np.complex128)

            for k, c in enumerate(coeffs):
                reg_back = Register(
                    num_qubits=self.n_qubits, density_matrix=self.density_matrix
                )
                reg_forw = Register(
                    num_qubits=self.n_qubits, density_matrix=self.density_matrix
                )

                params_back = self.thetas.copy()
                params_forw = self.thetas.copy()

                params_forw[i] += np.pi / 2  # (k + 1) * self.fd_step
                params_back[i] -= np.pi / 2  # (k + 1) * self.fd_step

                circ_back = self.build_circuit(thetas=params_back)
                circ_forw = self.build_circuit(thetas=params_forw)

                reg_back.apply_circuit(circ_back)
                reg_forw.apply_circuit(circ_forw)

                if self.density_matrix:
                    rho_i += reg_forw[:, :] - reg_back[:, :]
                    # rho_i += c * (
                    #     ((-1) ** (k)) * reg_forw[:, :]
                    #     + ((-1) ** (k + 1)) * reg_back[:, :]
                    # )
                else:
                    rho_i += reg_forw[:] - reg_back[:]
                    # rho_i += c * (
                    #     ((-1) ** (k)) * reg_forw[:] + ((-1) ** (k + 1)) * reg_back[:]
                    # )

            if self.density_matrix:
                diff_states[i, :, :] = rho_i / 2  # self.fd_step
            else:
                diff_states[i, :] = rho_i / 2  # self.fd_step

        return diff_states

    def covar(self, hamiltonian: Hamiltonian, n_iterations: int = 100):
        rho = self.run_circuit()
        loss = [hamiltonian.expectation(rho)]
        regularizer = self.regularization * np.identity(self.thetas.size)
        self.f_norm = []
        for iter in range(n_iterations):
            # run gradient descent for many iterations: then should solution be achievable in one step
            f = self.compute_f(hamiltonian)
            self.f_norm.append(np.linalg.norm(f))
            J = self.compute_jacobian(hamiltonian)
            J_inv = np.linalg.pinv((J.transpose() @ J) + regularizer) @ J.transpose()
            self.thetas -= self.learning_rate * (J_inv @ f).real
            rho = self.run_circuit()
            loss.append(hamiltonian.expectation(rho))
        return loss

    def sample_pool(self) -> list:
        pool = ["X", "Y", "Z"]
        n_local_max = 2
        O = []
        np.random.seed(self.seed)
        for _ in range(self.pool_size):
            locality = np.sum(
                np.random.uniform(low=0, high=1, size=(n_local_max,)) > 0.5
            )
            if locality > 0:
                qubits = np.random.choice(
                    range(self.n_qubits), size=(locality,), replace=False
                ).tolist()
                operators = np.random.choice(pool, size=(locality,))
                O_k = []
                for qubit, operator in zip(qubits, operators):
                    obs = getattr(unitaries, operator)
                    O_k.append(obs(qubit))
                O.append(O_k)
        self.N_c = len(O)
        return O

