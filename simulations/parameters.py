from .imports import *


@dataclass
class Parameters:
    seed: bool = 0
    n_qubits: int = 3
    n_layers: int = 10
    n_seeds: int = 10
    vqe: bool = False
    pool_size: int = 100
    shadow_size: int = int(1e5)
    n_chunks: int = 10
    savepth: str = os.getcwd() + "/results/"
    hamiltonian: str = "TFI-u"
    mixer: str = "X"
    ansatz: str = "sel"
    p_noise: float = 0.0
    density_matrix: bool = True
    add_scrambling: bool = True
    two_qubit_noise: bool = False
    dt: float = 0.1
    learning_rate: float = 6e-2
    regularization: float = 1e-3
    noise: str = "Depolarising"
    experiment: str = "test"  # folder name

    def __init__(self, kwargs: Dict = {}, mkdir: bool = False) -> None:
        self.update(kwargs)
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
        setattr(self, "savepth", f"{self.savepth}{self.experiment}/")
        if mkdir and not os.path.isdir(self.savepth):
            os.mkdir(self.savepth)
            self.save()

    def update(self, kwargs, save=False) -> None:
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found")
        if save:
            self.save()

    def save(self) -> None:
        json_dump = json.dumps(asdict(self))
        with open(self.savepth + "parameters.json", "w") as f:
            f.write(json_dump)
