# Import functions from the Ising_Model_Fast module
from Ising_Model_Fast import *
import time


# ===========================================
#       PARAMETERS AND CONFIGURATION
# ===========================================

# Global parameters
SEED = 42
np.random.seed(SEED)

test_verbose = True


# Paremeter configuration
N = 50
J1 = 0.5
J2 = 1.0
T = 0.2
MC_steps = 1_000_000

# TIME TESTING
t_inits_i = time.time()
# TIME TESTING

# ===========================================
#       INITIALIZATIONS AND SETUP
# ===========================================

# Initialize lattice
p = 0.5
lattice = np.random.choice([1, -1], size=(N, N), p=[p, 1-p])

# Get initial energy
initial_energy = get_energy(lattice, N, J1, J2)


# Setup simulation parameters

# nº of states to save (gif/images)
n_images = 100

# Set the MC steps idxs to save the images
img_spacing = np.unique(np.logspace(0, np.log10(MC_steps), n_images, endpoint=True, dtype=int))
n_images = len(img_spacing)

# Configure and save paths in a dict
paths = path_configuration(N, T)

# TIME TESTING
t_inits_f = time.time()
t_simul_i = time.time()
# TIME TESTING

# ===========================================
#               SIMULATION
# ===========================================

# Nice parameter formatting
simulation_params = {
    'MC_steps' : MC_steps,
    'T' : T,
    'N' : N,
    'J1' : J1,
    'J2' : J2,
    'save_images' : True,
    'image_spacing' : img_spacing,
    'verbose' : 0
    }


# Run the Metropolis algorithm
spins, energies, images, last_config = metropolis(lattice=lattice, energy=initial_energy, **simulation_params)

# TIME TESTING
t_simul_f = time.time()
t_save_data_i = time.time()
# TIME TESTING

# Save magnetization and energy data
for data, name in zip([spins, energies], ['spins.dat', 'energies.dat']):
    save_data(data, paths['data'], name)

# Save final configuration
save_data(last_config, paths['data'], 'final_configuration.dat')

# TIME TESTING
t_save_data_f = time.time()
t_save_gif_i = time.time()
# TIME TESTING

# Save gif
create_gif(images, save_dir=paths['figures'], filename='demo.gif', scale=5, fps=15, cmap='plasma')

# TIME TESTING
t_save_gif_f = time.time()
# TIME TESTING

# ===========================================
if test_verbose:
    print('\n===========================================')
    print("Time testing results:")
    print(f"\tInitialization time: {t_inits_f - t_inits_i:.2f} s")
    print(f"\tSimulation time: {t_simul_f - t_simul_i:.2f} s")
    print(f"\t\tFor N = {N} and MC_steps = {MC_steps}")
    print(f"\tData saving time: {t_save_data_f - t_save_data_i:.2f} s")
    print(f"\tGif saving time: {t_save_gif_f - t_save_gif_i:.2f} s")
    print("===========================================\n")
    