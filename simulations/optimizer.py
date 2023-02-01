from .imports import *
from .optimizers.covariance_root_finding import CovarianceRootFinding
from .optimizers.gradients import Gradients
from .optimizers.natural_gradients import NaturalGradients
from .parameters import Parameters


class Optimizer(Gradients, NaturalGradients, CovarianceRootFinding):
    def __init__(self, parameters: Parameters) -> None:
        CovarianceRootFinding.__init__(self, parameters)
        self.learning_rate = parameters.learning_rate
        self.regularization = parameters.regularization
        self.fd_step = 1e-6
        self.fd_coeffs = [
            [1 / 2],
            [2 / 3, -1 / 12],
            [3 / 4, -3 / 20, 1 / 60],
            [4 / 5, -1 / 5, 4 / 105, -1 / 280],
        ]
        # Finite difference coefficients taken from:
        # https://en.wikipedia.org/wiki/Finite_difference_coefficient

