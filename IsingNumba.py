import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
from scipy.ndimage import convolve
import timeit

def init_lattice(N, p):
    """
    Initialize a lattice of size N x N with random spins.
    Each spin is set to 1 with probability p and -1 with probability 1-p.
    """
    return np.random.choice([-1, 1], size=(N, N), p=[1-p, p])

def plot_lattice(lattice):
    """
    Plot the lattice using matplotlib.
    """
    fig, ax = plt.subplots()
    ax.imshow(lattice, cmap='coolwarm', interpolation='nearest')
    ax.set_title('Ising Model Lattice')
    ax.axis('off')
    plt.show()
    plt.close(fig)


def get_energy(lattice, J1=0.5, J2=1.0):
    """
    Calculate the energy of the lattice using periodic boundary conditions.
    The energy is calculated based on the nearest and next-nearest neighbors.
    """
    N = lattice.shape[0]
    # Nearest neighbors
    nn_kernel = np.array([
                        [0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0]])
    nn_energy = -J1 * lattice * convolve(lattice, nn_kernel, mode='wrap')
   
    # Next-nearest neighbors
    nnn_kernel = np.array([
                        [1, 0, 1],
                        [0, 0, 0],
                        [1, 0, 1]])
    nnn_energy = -J2 * lattice * convolve(lattice, nnn_kernel, mode='wrap')

    return np.sum(nn_energy + nnn_energy)/2


@njit
def get_energy_numba(lattice, J1=0.5, J2=1.0):
    """
    Calculate the energy of the lattice using periodic boundary conditions.
    The energy is calculated based on the nearest and next-nearest neighbors.
    This version uses Numba for JIT compilation.
    """
    N = lattice.shape[0]
    nn_energy = 0.0
    nnn_energy = 0.0

    for i in range(N):
        for j in range(N):
            # Nearest neighbors
            nn_energy += -J1 * lattice[i, j] * (lattice[(i + 1) % N, j] + lattice[i, (j + 1) % N])
            # Next-nearest neighbors
            nnn_energy += -J2 * lattice[i, j] * (lattice[(i + 1) % N, (j + 1) % N] + lattice[(i - 1) % N, (j - 1) % N])

    return nn_energy + nnn_energy


N = 50
p = 0.5
lattice = init_lattice(N, p)

energy = get_energy(lattice)
print("Energy (NumPy):", energy)

energy_numba = get_energy_numba(lattice)
print("Energy (Numba):", energy_numba)



print("\nTime comparison:")

numba_time = timeit.timeit(lambda: get_energy_numba(lattice), number=1000)
print("Numba time:", numba_time)

numpy_time = timeit.timeit(lambda: get_energy(lattice), number=1000)
print("NumPy time:", numpy_time)
