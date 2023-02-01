from ..hamiltonian import Hamiltonian
from ..imports import *


class NaturalGradients(object):
    def compute_Fc_ij(self) -> np.ndarray:
        n_parameters = self.thetas.size
        Fc = np.full((n_parameters, n_parameters), np.nan)
        diff_states = np.full(
            (n_parameters, 2 ** self.n_qubits), np.nan, dtype=np.complex_
        )
        for i in range(n_parameters):
            reg = Register(num_qubits=self.n_qubits, density_matrix=self.density_matrix)
            circ = Circuit(self.build_ansatz(self.thetas, dU_di=i))
            reg.apply_circuit(circ)
            diff_states[i, :] = (-1j / 2) * reg[:]

        psi = self.run_circuit().reshape(-1, 1)
        for i in range(n_parameters):
            for j in range(n_parameters):
                if j >= i:
                    Fc[i, j] = (
                        diff_states[i, :].conjugate().transpose()
                        @ diff_states[j, :]
                        # - (diff_states[i, :].conjugate().transpose() @ psi)
                        # * (psi.conjugate().transpose() @ diff_states[j, :])
                    ).real
                    Fc[j, i] = Fc[i, j]
        return Fc

    def compute_Fq_ij(self):
        n_parameters = self.thetas.size
        Fq = np.full((n_parameters, n_parameters), np.nan)
        diff_states = np.full(
            (n_parameters, 2 ** self.n_qubits, 2 ** self.n_qubits),
            np.nan,
            dtype=np.complex_,
        )
        coeffs = self.fd_coeffs[0]
        for i in range(n_parameters):
            rho_i = np.zeros(
                (2 ** self.n_qubits, 2 ** self.n_qubits), dtype=np.complex128
            )
            for k, c in enumerate(coeffs):
                reg_back = Register(
                    num_qubits=self.n_qubits, density_matrix=self.density_matrix
                )
                reg_forw = Register(
                    num_qubits=self.n_qubits, density_matrix=self.density_matrix
                )

                params_back = self.thetas.copy()
                params_forw = self.thetas.copy()

                params_forw[i] += (k + 1) * self.fd_step
                params_back[i] -= (k + 1) * self.fd_step

                circ_back = self.build_circuit(thetas=params_back)
                circ_forw = self.build_circuit(thetas=params_forw)

                reg_back.apply_circuit(circ_back)
                reg_forw.apply_circuit(circ_forw)
                rho_i += c * (
                    ((-1) ** (k)) * reg_forw[:, :] + ((-1) ** (k + 1)) * reg_back[:, :]
                )

            diff_states[i, :, :] = rho_i / self.fd_step

        for i in range(n_parameters):
            for j in range(n_parameters):
                if j >= i:
                    Fq[i, j] = np.trace(
                        diff_states[i, :, :] @ diff_states[j, :, :]
                    ).real
                    Fq[j, i] = Fq[i, j]
        return Fq

    def compute_y(self, hamiltonian: Hamiltonian):
        n_parameters = self.thetas.size
        y = np.full(self.thetas.shape, np.nan)
        rho = self.run_circuit()
        H_dot_rho = hamiltonian.H_mat @ rho
        coeffs = self.fd_coeffs[0]
        for i in range(n_parameters):
            rho_i = np.zeros(
                (2 ** self.n_qubits, 2 ** self.n_qubits), dtype=np.complex128
            )
            for k, c in enumerate(coeffs):
                reg_back = Register(
                    num_qubits=self.n_qubits, density_matrix=self.density_matrix
                )
                reg_forw = Register(
                    num_qubits=self.n_qubits, density_matrix=self.density_matrix
                )

                params_back = self.thetas.copy()
                params_forw = self.thetas.copy()

                params_forw[i] += (k + 1) * self.fd_step
                params_back[i] -= (k + 1) * self.fd_step

                circ_back = self.build_circuit(thetas=params_back)
                circ_forw = self.build_circuit(thetas=params_forw)

                reg_back.apply_circuit(circ_back)
                reg_forw.apply_circuit(circ_forw)
                rho_i += c * (
                    ((-1) ** (k)) * reg_forw[:, :] + ((-1) ** (k + 1)) * reg_back[:, :]
                )

            rho_i /= self.fd_step
            y[i] = np.trace(rho_i @ H_dot_rho).real

        return y

    def imaginary_time_evolution_pure(
        self, hamiltonian: Hamiltonian, n_iterations: int = 30
    ):
        assert not self.density_matrix
        rho = self.run_circuit()
        loss = [hamiltonian.expectation(rho)]
        regularizer = self.regularization * np.identity(self.thetas.size)
        for _ in range(n_iterations):
            fisher = self.compute_Fc_ij()
            F_inv = np.linalg.pinv(fisher + regularizer)
            g = self.compute_gradients(hamiltonian.expectation)
            self.thetas -= self.learning_rate * (F_inv @ g)
            rho = self.run_circuit()
            loss.append(hamiltonian.expectation(rho))
        return loss

    def imaginary_time_evolution_mixed(
        self, hamiltonian: Hamiltonian, n_iterations: int = 30
    ):
        assert self.density_matrix
        rho = self.run_circuit()
        loss = [hamiltonian.expectation(rho)]
        regularizer = self.regularization * np.identity(self.thetas.size)
        for _ in range(n_iterations):
            Fq = self.compute_Fq_ij()
            F_inv = np.linalg.pinv(Fq + regularizer)
            y = self.compute_y(hamiltonian)
            self.thetas -= self.learning_rate * (F_inv @ y)
            rho = self.run_circuit()
            loss.append(hamiltonian.expectation(rho))
        return loss

    def quantum_natural_gradient_descent(
        self, hamiltonian: Hamiltonian, n_iterations: int = 30
    ):
        assert self.density_matrix
        rho = self.run_circuit()
        loss = [hamiltonian.expectation(rho)]
        regularizer = self.regularization * np.identity(self.thetas.size)
        for _ in range(n_iterations):
            Fq = self.compute_Fq_ij()
            F_inv = np.linalg.pinv(Fq + regularizer)
            g = self.compute_gradients(hamiltonian.expectation)
            self.thetas -= self.learning_rate * (F_inv @ g)
            rho = self.run_circuit()
            loss.append(hamiltonian.expectation(rho))
        return loss

    def imaginary_time_evolution(
        self, hamiltonian: Hamiltonian, n_iterations: int = 30
    ):
        if self.density_matrix:
            return self.imaginary_time_evolution_mixed(hamiltonian, n_iterations)
        else:
            return self.imaginary_time_evolution_pure(hamiltonian, n_iterations)
