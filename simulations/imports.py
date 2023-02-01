import sys
import os
import numpy as np
import time
import json
from dataclasses import asdict, dataclass
import itertools
import inspect
import shutil
from IPython.display import clear_output
from tqdm.notebook import tqdm
from datetime import datetime
import json
import importlib
from typing import Any, List, Optional, Tuple, Union, Dict
import pickle
import sympy
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import random
from random import choice
from string import ascii_uppercase
from copy import deepcopy

# Cirq and tensorflow quantum
import cirq

# from cirq import Simulator, DensityMatrixSimulator
# import tensorflow as tf
# import tensorflow_quantum as tfq

# PyQuEST
from pyquest import decoherence
from pyquest import Register
from pyquest import unitaries
from pyquest.unitaries import U, H, S, Rz, Rx, Ry, X, Y, Z, R, PauliProduct
from pyquest import Circuit
from pyquest.gates import M
from pyquest.operators import MatrixOperator
from .pyquest_multigates import *

from sklearn.linear_model import LinearRegression as LR
from scipy.optimize import curve_fit

# from qiskit.visualization import plot_state_city, plot_bloch_multivector
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["font.size"] = 20
matplotlib.rcParams["figure.figsize"] = (10, 6)
matplotlib.rcParams["savefig.bbox"] = "tight"
