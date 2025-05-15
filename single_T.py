# Import functions from the Ising_Model_Fast module
from Ising_Model_Fast import *
import numpy as np
import matplotlib.pyplot as plt


# ===========================================
#       PARAMETERS AND CONFIGURATION
# ===========================================
seed = 3141
np.random.seed(seed)

# Simulation parameters for single-T run (images + time evolution)
N = 100
J1, J2 = 1.0, 0.0
T_fixed = 4.0
MC_steps = 5_000
Iterations = N*N*MC_steps


# ===========================================
#       INITIALIZATIONS AND SETUP
# ===========================================
# Lattice and energy for single-T run
tilattice = initialize_lattice(N, p=0.60, seed=seed)
initial_energy = get_energy(tilattice, N, J1, J2)
# Paths for saving
paths = path_configuration(N, T_fixed)


# ===========================================
#         SINGLE-T SIMULATION
# ===========================================
simulation_params = {
    'MC_steps': Iterations,
    'T': T_fixed,
    'N': N,
    'J1': J1,
    'J2': J2,
    'save_images': True,
    'images_spacing': np.unique(np.logspace(0, np.log10(MC_steps), 100, dtype=int)),
    'seed': seed,
}
# Run Metropolis once for images, spins and energies
spins, energies, images, last_config = metropolis(
    lattice=tilattice,
    energy=initial_energy,
    **simulation_params
)


use_last = 10000 if Iterations > 10000 else int(0.5 * Iterations)

mean_M = np.mean(spins[-use_last:])
std_M = np.std(spins[-use_last:], ddof=1)
mean_E = np.mean(energies[-use_last:])
std_E = np.std(energies[-use_last:], ddof=1)
C_v = np.var(energies[-use_last:], ddof=1) / (T_fixed**2 * N * N)
std_C = std_E / (T_fixed**2 * N * N)


with open(
    paths['data'] / f'single_T_statistics_N{N}_T{T_fixed}.txt',
    'w', encoding='utf-8'
) as f:
    f.write(f"# Single temperature analysis for T = {T_fixed}, N = {N}\n")
    f.write("Quantity\tMean\tStdError\n")
    f.write(f"Magnetization\t{mean_M:.6f}\t{std_M:.6f}\n")
    f.write(f"Energy\t{mean_E:.6f}\t{std_E:.6f}\n")
    f.write(f"Specific Heat\t{C_v:.6f}\t{std_C:.6f}\n")

# Save time-series data and images
fast_save_data(spins, paths['data'], 'spins')
fast_save_data(energies, paths['data'], 'energies')
fast_save_data(last_config, paths['data'], f'last_config_N{N}_T{T_fixed}_MC{MC_steps}')
create_gif(images, save_dir=paths['figures'], filename='demo.gif', scale=5, fps=15, cmap='plasma')
save_images_as_png(images, save_dir=paths['images'], prefix='ising', cmap='plasma', scale=5)


# Plot time-series
def plot_time_series(data, ylabel, title, fname):
    plt.figure()
    plt.plot(data)
    plt.xlabel('Paso Monte Carlo')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(paths['figures'] / fname)
    plt.close()

plot_time_series(spins, 'Magnetization', 'Magnetization vs MC steps', 'magnetizacion_vs_pasos.png')
plot_time_series(energies, 'Energy', 'Energy vs MC steps', 'energia_vs_pasos.png')