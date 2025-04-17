import numba
from numba import njit
import numpy as np


@njit
def calcular_probabilidad_optimized(lattice, i, j, N, J1, J2, T):
    s = lattice[i, j]
    
    # Nearest neighbors (wrap around using modulo)
    nn_sum = (
        lattice[(i + 1) % N, j] +
        lattice[(i - 1) % N, j] +
        lattice[i, (j + 1) % N] +
        lattice[i, (j - 1) % N]
    )
    
    # Next-nearest neighbors (diagonals)
    nnn_sum = (
        lattice[(i + 1) % N, (j + 1) % N] +
        lattice[(i + 1) % N, (j - 1) % N] +
        lattice[(i - 1) % N, (j + 1) % N] +
        lattice[(i - 1) % N, (j - 1) % N]
    )
    
    # Energy change for flipping s
    dE = 2 * s * (J1 * nn_sum + J2 * nnn_sum)
    
    # Metropolis criterion — always accept if dE <= 0
    if dE <= 0:
        return 1.0
    else:
        return np.exp(-dE / T)
