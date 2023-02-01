from ..hamiltonian import Hamiltonian
from ..imports import *


class Gradients(object):
    def compute_gradients(self, loss_func, s: float = np.pi / 2):
        grads = dict(self.thetas)
        for name, val in self.thetas.items():
            reg_back = Register(
                num_qubits=self.n_qubits, density_matrix=self.density_matrix
            )
            reg_forw = Register(
                num_qubits=self.n_qubits, density_matrix=self.density_matrix
            )

            params_back = dict(self.thetas)
            params_forw = dict(self.thetas)

            params_back[name] = val - s
            params_forw[name] = val + s

            circ_back = self.build_circuit(thetas=params_back)
            circ_forw = self.build_circuit(thetas=params_forw)
            reg_back.apply_circuit(circ_back)
            reg_forw.apply_circuit(circ_forw)

            state_back = reg_back[:, :] if self.density_matrix else reg_back[:]
            state_forw = reg_forw[:, :] if self.density_matrix else reg_forw[:]

            f_back = loss_func(state_back)
            f_forw = loss_func(state_forw)

            grads[name] = (f_forw - f_back) / 2

        return grads

    def gradient_descent(self, hamiltonian: Hamiltonian, n_iterations: int = 30):
        rho = self.run_circuit()
        loss = [hamiltonian.expectation(rho)]
        for _ in range(n_iterations):
            gradients = self.compute_gradients(hamiltonian.expectation)
            for key, grad_val in gradients.items():
                self.thetas[key] -= self.learning_rate * grad_val
            rho = self.run_circuit()
            loss.append(hamiltonian.expectation(rho))
        return loss

