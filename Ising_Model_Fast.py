# Imports
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit,prange
from scipy.ndimage import convolve
import imageio
from pathlib import Path

# ----------------------------
#   Function definitions
# ----------------------------

def initialize_lattice(N, p=0.5, seed=None, **kwargs):
    """
    Initialize a 2D lattice of size N x N with random spins (+1 or -1).
    Parameters:
    -----------
    - N : int
        Size of the lattice (N x N).
    - p : float, optional
        Probability of a spin being +1 (default is 0.5).
    - seed : int, optional
        Random seed for reproducibility (default is None).
    Returns:
    --------
    - numpy.ndarray
        A 2D array representing the initialized lattice.
    
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    web = np.random.choice([1, -1], size=(N, N), p=[p, 1-p])
    return web.astype(np.int8)  # Convert to int8 for memory efficiency

def get_energy(lattice, N, J1, J2, **kwargs):
    """
    Calculate the total energy of a 2D Ising model lattice with nearest-neighbor (NN) 
    and next-nearest-neighbor (NNN) interactions.
    Parameters:
    -----------
    - lattice : numpy.ndarray
        A 2D array representing the spin configuration of the lattice. Each element 
        is typically +1 or -1, representing spin states.
    - N : int
        Side length of the square lattice .
    - J1 : float
        The interaction strength for nearest-neighbor (NN) interactions.
    - J2 : float
        The interaction strength for next-nearest-neighbor (NNN) interactions.
    Returns:
    --------
    - float
        The total energy of the lattice.
    """

    kernel_nn = np.array([
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0]])
    
    kernel_nnn = np.array([
                [1, 0, 1],
                [0, 0, 0],
                [1, 0, 1]])
    
    energy_nn = -J1 * lattice * convolve(lattice, kernel_nn, mode='wrap')
    energy_nnn = -J2 * lattice * convolve(lattice, kernel_nnn, mode='wrap')
    
    return (energy_nn + energy_nnn).sum()/2

@njit
def get_energy_fast(lattice, N, J1, J2):
    """
    Calculate the total energy of a 2D Ising model lattice with nearest-neighbor (NN) 
    and next-nearest-neighbor (NNN) interactions.
    Parameters:
    -----------
    - lattice : numpy.ndarray
        A 2D array representing the spin configuration of the lattice. Each element 
        is typically +1 or -1, representing spin states.
    - N : int
        Side length of the square lattice .
    - J1 : float
        The interaction strength for nearest-neighbor (NN) interactions.
    - J2 : float
        The interaction strength for next-nearest-neighbor (NNN) interactions.
    Returns:
    --------
    - float
        The total energy of the lattice.
    """
    
    energy = 0.0

    for i in range(N):
        for j in range(N):
            # Nearest neighbors
            energy += -J1 * lattice[i, j] * (
                lattice[(i+1)%N, j] + lattice[i, (j+1)%N] +
                lattice[(i-1)%N, j] + lattice[i, (j-1)%N]
            )
            
            # Next nearest neighbors
            energy += -J2 * lattice[i, j] * (
                lattice[(i+1)%N, (j+1)%N] + lattice[(i-1)%N, (j-1)%N] +
                lattice[(i+1)%N, (j-1)%N] + lattice[(i-1)%N, (j+1)%N]
            )

    return energy / 2.0

@njit
def compute_specific_heat(energy_array, N, T, burn_in=0.5, **kwargs):
    """
    Compute the specific heat per spin of the Ising model system.
    
    Parameters:
    -----------
    - energy_array : numpy.ndarray
        Array of energy values recorded at each Monte Carlo step.
    - N : int
        Lattice side length (for total number of spins = N*N).
    - T : float
        Temperature of the system.
        
    Returns:
    --------
    - float
        Specific heat per spin.
    """
    burn_in_index = int(len(energy_array) * burn_in)
    C = np.var(energy_array[burn_in_index:]) / (T**2 * N*N)

    return C


@njit
def get_dE(lattice, x, y, N, J1, J2):
    """
    Calculate the change in energy (ΔE) for flipping a spin in a 2D Ising model.
    This function computes the energy difference that would result from flipping
    the spin at position (x, y) in the lattice. The calculation considers both
    nearest-neighbor (NN) and next-nearest-neighbor (NNN) interactions.
    - Parameters:
    ----------
        - lattice (numpy.ndarray): A 2D array representing the spin lattice, where
                                 each element is either +1 or -1.
        - x (int): The x-coordinate of the spin to be flipped.
        - y (int): The y-coordinate of the spin to be flipped.
        - N (int): The side length of the lattice (assumed to be NxN and periodic).
        - J1 (float): The interaction strength for nearest neighbors.
        - J2 (float): The interaction strength for next-nearest neighbors.
    - Returns:
        - float: The change in energy (ΔE) resulting from flipping the spin at (x, y).
    """

    nn_sum = (
        lattice[(x-1)%N, y] + lattice[(x+1)%N, y] +
        lattice[x, (y-1)%N] + lattice[x, (y+1)%N]
    )

    nnn_sum = (
        lattice[(x-1)%N, (y-1)%N] + lattice[(x+1)%N, (y-1)%N] +
        lattice[(x-1)%N, (y+1)%N] + lattice[(x+1)%N, (y+1)%N]
    )

    dE = 2 * lattice[x, y] * (J1 * nn_sum + J2 * nnn_sum)

    return dE


@njit
def metropolis(lattice, MC_steps, T, energy, N, J1, J2, seed=42, save_images=False, images_spacing=np.array([0, 1])):
    """
    Perform the Metropolis algorithm for simulating the Ising model.
    Parameters:
    -----------
    - lattice : numpy.ndarray
        Initial NxN lattice configuration of spins (+1 or -1).
    - MC_steps : int
        Number of Monte Carlo steps to perform.
    - T : float
        Temperature of the system.
    - energy : float
        Initial energy of the system.
    - N : int
        Size of the lattice (NxN).
    - J1 : float
        Interaction strength for nearest neighbors.
    - J2 : float
        Interaction strength for next-nearest neighbors.
    - save_images : bool, optional
        Whether to save snapshots of the lattice during the simulation (default is False).
    - images_spacing : list of int, optional
        List of Monte Carlo steps at which to save lattice snapshots (default is numpy.array [0,1]).
.
    Returns:
    --------
    - net_spins : numpy.ndarray
        Array of net magnetization values at each Monte Carlo step.
    - net_energy : numpy.ndarray
        Array of energy values at each Monte Carlo step.
    - images : numpy.ndarray or None
        Array of saved lattice snapshots if `save_images` is True, otherwise -1.
    - last_config : numpy.ndarray
        Final lattice configuration after the simulation.
    Notes:
    ------
    - The Metropolis algorithm is used to simulate the evolution of the Ising model.
    - The flipping condition is determined by the change in energy (dE) and the temperature (T).
    - If `save_images` is True, the lattice snapshots are saved at the specified `images_spacing` steps.
    - If the system is not in equilibrium after the simulation, it is possible to run again using the returned last lattice configuration as the new initial state. 
    """

    np.random.seed(seed)  # Set the random seed for reproducibility

    # 1. Initialize variables
    web = lattice.copy()
    net_spins = np.zeros(MC_steps)
    net_energy = np.zeros(MC_steps)

    #------------------------
    #   Image saving logic
    #------------------------
    aux_img_idx = 0
    if save_images and images_spacing is not None:
        images = np.empty((len(images_spacing), N, N), dtype=np.int8)
        
    # 'None' used for consistency in the return statement
    else:
        images = np.zeros((1, N, N), dtype=np.int8)  # Placeholder for images


    # ---------------------
    #       Main loop
    # ---------------------
    for t in range(MC_steps):
        if save_images and t in images_spacing:
            images[aux_img_idx] = web.copy()
            aux_img_idx += 1

        # 2. Choose a random spin to evaluate
        x = np.random.randint(0, N)
        y = np.random.randint(0, N)


        # 3. Compute the change in energy
        dE = get_dE(web, x, y, N, J1, J2)

        # 4. Apply flipping condition
        if ((dE > 0) * (np.random.random() < np.exp(-dE/T))):
            web[x,y] *= -1
            energy += dE
        elif dE<=0:
            web[x,y] *= -1
            energy += dE
            
        # 5. Save net spin (magnetization) and energy
        net_spins[t] = web.sum()/(N**2)
        net_energy[t] = energy

        if save_images:
            images[-1] = web.copy()
    
    last_config = web.copy()

    return net_spins, net_energy, images, last_config


@njit(parallel=False)
def metropolis_large(lattice, MC_steps, T, energy, N, J1, J2, seed=42):
    """
    Perform the Metropolis algorithm for simulating the Ising model.
    Parameters:
    -----------
    - lattice : numpy.ndarray
        Initial NxN lattice configuration of spins (+1 or -1).
    - MC_steps : int
        Number of Monte Carlo steps to perform.
    - T : float
        Temperature of the system.
    - energy : float
        Initial energy of the system.
    - N : int
        Size of the lattice (NxN).
    - J1 : float
        Interaction strength for nearest neighbors.
    - J2 : float
        Interaction strength for next-nearest neighbors.
    Returns:
    --------
    - net_spins : numpy.ndarray
        Array of net magnetization values at each Monte Carlo step.
    - net_energy : numpy.ndarray
        Array of energy values at each Monte Carlo step.
    - last_config : numpy.ndarray
        Final lattice configuration after the simulation.
    Notes:
    ------
    - The Metropolis algorithm is used to simulate the evolution of the Ising model.
    - The flipping condition is determined by the change in energy (dE) and the temperature (T).
    - If `save_images` is True, the lattice snapshots are saved at the specified `images_spacing` steps.
    - If the system is not in equilibrium after the simulation, it is possible to run again using the returned last lattice configuration as the new initial state. 
    """

    # Select a seed for reproducibility
    np.random.seed(seed)

    # 1. Initialize variables
    web = lattice.copy()
    net_spins = np.empty(MC_steps, dtype=np.float32)            # Updated every MC step
    net_energy = np.empty(MC_steps, dtype=np.float32)           # Updated every MC step
    N_squared = N*N         
    
    energy = get_energy_fast(web, N, J1, J2)  # Initial energy
    # =============================================
    #               Main loop
    # =============================================
    for t in range(0, MC_steps):
        # Save magnetization at every MC step
        net_spins[t] = web.sum()/(N**2)
        net_energy[t] = energy
        
        # x_idx = np.random.randint(0, N, size=N_squared)
        # y_idx = np.random.randint(0, N, size=N_squared)

        for k in range(N_squared):
            # 2. Choose a random spin to evaluate
            x = np.random.randint(0, N)
            y = np.random.randint(0, N)

            # 3. Compute the change in energy
            dE = get_dE(web, x, y, N, J1, J2)

            # 4. Apply flipping condition
            if ((dE > 0) * (np.random.random() < np.exp(-dE/T))):
                web[x,y] *= -1
                energy += dE
            elif dE<=0:
                web[x,y] *= -1
                energy += dE

    return net_spins, net_energy, web.copy()
    

@njit(parallel=False)
def metropolis_large_opt(lattice, MC_steps, T, energy, N, J1, J2, seed=42):
    """
    *WORK IN PROGRESS*
    ------------------
    Perform the Metropolis algorithm for simulating the Ising model.
    Parameters:
    -----------
    - lattice : numpy.ndarray
        Initial NxN lattice configuration of spins (+1 or -1).
    - MC_steps : int
        Number of Monte Carlo steps to perform.
    - T : float
        Temperature of the system.
    - energy : float
        Initial energy of the system.
    - N : int
        Size of the lattice (NxN).
    - J1 : float
        Interaction strength for nearest neighbors.
    - J2 : float
        Interaction strength for next-nearest neighbors.
    Returns:
    --------
    - net_spins : numpy.ndarray
        Array of net magnetization values at each Monte Carlo step.
    - net_energy : numpy.ndarray
        Array of energy values at each Monte Carlo step.
    - last_config : numpy.ndarray
        Final lattice configuration after the simulation.
    Notes:
    ------
    - The Metropolis algorithm is used to simulate the evolution of the Ising model.
    - The flipping condition is determined by the change in energy (dE) and the temperature (T).
    - If `save_images` is True, the lattice snapshots are saved at the specified `images_spacing` steps.
    - If the system is not in equilibrium after the simulation, it is possible to run again using the returned last lattice configuration as the new initial state. 
    """

    # Select a seed for reproducibility
    np.random.seed(seed)

    # 1. Initialize variables
    web = lattice.copy()
    net_spins = np.empty(MC_steps, dtype=np.float32)            # Updated every MC step
    net_energy = np.empty(MC_steps, dtype=np.float32)           # Updated every MC step
    N_squared = N*N         
    
    energy = get_energy_fast(web, N, J1, J2)  # Initial energy
    # =============================================
    #               Main loop
    # =============================================
    for t in range(0, MC_steps):
        # Save magnetization at every MC step
        net_spins[t] = web.sum()/(N_squared)
        net_energy[t] = energy
        
        x_idx = np.random.randint(0, N, size=N_squared)
        y_idx = np.random.randint(0, N, size=N_squared)
        acceptance = np.random.random(N_squared)

        for k in range(N_squared):
            # 2. Choose a random spin to evaluate
            x = x_idx[k]
            y = y_idx[k]

            # 3. Compute the change in energy
            dE = get_dE(web, x, y, N, J1, J2)

            # 4. Apply flipping condition
            if ((dE > 0) * (acceptance[k] < np.exp(-dE/T))):
                web[x,y] *= -1
                energy += dE
            elif dE<=0:
                web[x,y] *= -1
                energy += dE

    return net_spins, net_energy, web.copy()
 

def path_configuration(N, T, J1=None, J2=None, simulations_dir='Simulations', data_dir='data', figures_dir='figures', images_dir='images', verbose=0):
    """
    Creates a directory structure for storing simulation data and figures.
    Parameters:
    -----------
    N : int
        The size of the simulation grid.
    T : float
        The temperature parameter for the simulation.
    J1 : float, optional
        The first coupling constant (default is None).
    J2 : float, optional
        The second coupling constant (default is None).
    simulations_dir : str, optional
        The name of the parent directory for simulations (default is 'Simulations').
    data_dir : str, optional
        The name of the subdirectory for storing data (default is 'data').
    figures_dir : str, optional
        The name of the subdirectory for storing figures (default is 'figures').
    images_dir : str, optional
        The name of the subdirectory for storing images (default is 'images').
    verbose : int, optional
        The verbosity level for logging messages:
        - 0: No output.
        - 1: Basic output.
        - 2: Detailed output (default is 0).
    Behavior:
    ---------
    - Creates a parent directory named based on the simulation parameters (N, T, J1, J2).
    - Creates subdirectories for data and figures within the parent directory.
    - Ensures that all directories are created if they do not already exist.
    Returns:
    --------
    Dictionary with the paths to the created directories.
    """
    
    if verbose > 0:
        print(f"Creating directory structure for N={N}, T={T}")

    # Create the main simulations directory if it doesn't exist
    simulations_dir = Path(simulations_dir)
    simulations_dir.mkdir(parents=True, exist_ok=True)
    
    # Parent folder
    if verbose > 1:
        print(f"Creating parent folder")
    if J1 is None or J2 is None:
        parent_name = f"Simulation_N{N}_T{T}"
    else:
        parent_name = f"Simulation_N{N}_T{T}_J1{J1}_J2{J2}"
    parent_dir = simulations_dir / parent_name
    parent_dir.mkdir(parents=True, exist_ok=True)

    # Sub folders
    data_dir = parent_dir / data_dir
    figures_dir = parent_dir / figures_dir
    images_dir = parent_dir / images_dir
    if verbose > 1:
        print(f"Creating data folder")
        print(f"Creating figures folder")
        print(f"Creating images folder")
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    paths_dict = {
        "parent": parent_dir,
        "data": data_dir,
        "figures": figures_dir,
        "images": images_dir
    }

    return paths_dict


def create_gif(images, save_dir, filename="simulation.gif", fps=10, scale=1, cmap="gray", color_map=None, verbose=True):
    """
    Create a GIF from a list of 2D numpy arrays.

    Parameters:
    - images: list of 2D numpy arrays (values in -1 or 1)
    - filename: output GIF filename
    - fps: frames per second
    - scale: scaling factor for image size (integer)
    - cmap: matplotlib colormap name (e.g., 'gray', 'viridis', 'plasma', etc.)
    """
    duration = len(images) / fps

    # Create writer
    file_path = save_dir / filename
    with imageio.get_writer(file_path, mode="I", duration=duration / len(images)) as writer:
        for img in images:
            # Normalize lattice values to 0-255
            norm_img = ((img + 1) * 127.5).astype(np.uint8)

            # Apply colormap if not grayscale
            if cmap == 'custom':
                colored_img = color_map
                colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
            elif cmap != "gray":
                colored_img = plt.get_cmap(cmap)(norm_img / 255.0)  # RGBA values 0-1
                colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)  # Drop alpha
            else:
                colored_img = np.stack([norm_img]*3, axis=-1)  # Grayscale to RGB

            # Scale image if needed
            if scale > 1:
                colored_img = colored_img.repeat(scale, axis=0).repeat(scale, axis=1)

            # Write frame
            writer.append_data(colored_img)
    if verbose > 0:
        print(f"GIF saved as {file_path}")


def save_data(data, save_dir, filename="data.dat", header=None, fmt='%.6f', verbose=0):

    file_path = save_dir / filename
    if header is None:
        np.savetxt(file_path, data, fmt=fmt)
    else:
        np.savetxt(file_path, data, header=header, fmt=fmt)
    if verbose > 0:
        print(f"{filename[:-3]} saved at {file_path}")


def fast_save_data(data, save_dir, filename="data", verbose=0):
    """
    Save data in binary format using numpy's .npy format.
    ----------
    Parameters:
        - data : numpy.ndarray
            The data to be saved.
        - save_dir : str or Path
            Directory where the file will be saved.
        - filename : str, optional
            Name of the output file (default is "data.npy").
        - verbose : int, optional
            Verbosity level for logging information (default is 0).
    """
    
    file_path = save_dir / filename
    np.save(file_path, data)
    if verbose > 0:
        print(f"{filename} saved at {file_path} in binary format (.npy)")



def save_energy_data(energy, save_dir, filename="energy.dat", verbose=0):
    
    file_path = save_dir / filename
    np.savetxt(file_path, energy, header="Energy values", fmt='%.6f')
    if verbose > 0:
        print(f"Energy data saved at {file_path}")


def save_magnetization_data(magnetization, save_dir, filename="magnetization.dat", verbose=0):
        
        file_path = save_dir / filename
        np.savetxt(file_path, magnetization, header="Magnetization values", fmt='%.6f')
        if verbose > 0:
            print(f"Magnetization data saved at {file_path}")


def save_lattice_data(lattice, save_dir, filename="lattice.dat", verbose=0):
    """
    Save the lattice configuration to a file.
    Parameters:
    -----------
    - lattice : numpy.ndarray
        The lattice configuration to save.
    - data_dir : str
        Directory where the file will be saved.
    - filename : str, optional
        Name of the output file (default is "lattice.dat").
    - verbose : int, optional
        Verbosity level for logging information (default is 0).
    """
    
    file_path = save_dir / filename
    np.savetxt(file_path, lattice, header="Lattice configuration", fmt='%d')
    if verbose > 0:
        print(f"Lattice configuration saved at {file_path}")

def get_M_E_C_of_T(lattice, energy, Ts, simulation_params, use_last=1000, burn_in=0.3):
    """
    Calculate the average magnetization, energy, and heat capacity for a range of temperatures.
    """
    N = simulation_params['N']
    avg_mags = np.empty(len(Ts), dtype=np.float32)
    avg_energies = np.empty(len(Ts), dtype=np.float32)
    heat_capacities = np.empty(len(Ts), dtype=np.float32)

    for i, T in enumerate(Ts):
        simulation_params['T'] = T
        print("="*20)
        print(f"Starting Simulation for T = {T:.2f}")
        print("="*20)

        mags, energies, _, _ = metropolis(
            lattice=lattice, energy=energy, **simulation_params
        )

        avg_mags[i] = np.mean(mags[-use_last:])
        avg_energies[i] = np.mean(energies[-use_last:])
        heat_capacities[i] = compute_specific_heat(
            energies[-use_last:], N, T, burn_in=burn_in
        )

    return avg_mags, avg_energies, heat_capacities

@njit
def std_manual(arr, mean_val):
    n = len(arr)
    return np.sqrt(np.sum((arr - mean_val) ** 2) / (n - 1))

@njit(parallel=True)
def get_M_E_C_of_T_numba(lattice, energy, Ts, N, J1, J2, MC_steps, seed, use_last=1000):
    n_temps = len(Ts)
    avg_mags = np.empty(n_temps, dtype=np.float32)
    std_mags = np.empty(n_temps, dtype=np.float32)
    avg_energies = np.empty(n_temps, dtype=np.float32)
    std_energies = np.empty(n_temps, dtype=np.float32)
    heat_capacities = np.empty(n_temps, dtype=np.float32)
    std_Cv = np.empty(n_temps, dtype=np.float32)
    susceptibilities = np.empty(n_temps, dtype=np.float32)
    std_chi = np.empty(n_temps, dtype=np.float32)
    cfe = 2
    cfm = np.sqrt(12)

    for i in prange(n_temps):
        T = Ts[i]
        local_lattice = lattice.copy()
        local_seed = seed + i * 1000
        mags, energies, _ = metropolis_large(
            local_lattice, MC_steps, T, energy, N, J1, J2, local_seed
        )

        e_sample = energies[-use_last:]
        m_sample = mags[-use_last:]

        mean_E = np.mean(e_sample)
        mean_M = np.mean(m_sample)

        std_E = std_manual(e_sample, mean_E)/ np.sqrt(use_last)
        std_M = std_manual(m_sample, mean_M)/ np.sqrt(use_last)

        # Calor específico
        C_v = np.sum((e_sample - mean_E)**2) / ((use_last - 1) * T**2 * N * N)

        # Error más realista del calor específico
        Cv_terms = (e_sample - mean_E)**2 / (T**2 * N * N)
        mean_Cv = np.mean(Cv_terms)
        std_C = std_manual(Cv_terms, mean_Cv) / np.sqrt(use_last)
        M2 = np.mean(m_sample**2)
        chi = (M2 - mean_M**2) / (T * N * N)
        
        # Para el error de chi (usando la varianza de M^2)
        chi_terms = (m_sample**2 - M2)**2 / (T**2 * N**4)
        mean_chi = chi
        std_chi_val = std_manual(chi_terms, np.mean(chi_terms)) / np.sqrt(use_last)

        # Guardar resultados
        avg_energies[i] = mean_E
        std_energies[i] = std_E*cfe
        avg_mags[i] = np.abs(mean_M)
        std_mags[i] = std_M*cfm
        heat_capacities[i] = C_v
        std_Cv[i] = std_C*cfe
        susceptibilities[i] = chi
        std_chi[i] = std_chi_val*cfm


    return avg_mags, std_mags, avg_energies, std_energies, heat_capacities, std_Cv, susceptibilities, std_chi



def save_images_as_png(images, save_dir, prefix="img", cmap="plasma", scale=1, verbose=True):
    """
    Save a list of 2D numpy arrays as PNG images.
    """
    for i, img in enumerate(images):
        norm_img = ((img + 1) * 127.5).astype(np.uint8)
        
        # Crear la imagen en RGB con el colormap
        if cmap != "gray":
            colored_img = plt.get_cmap(cmap)(norm_img / 255.0)
            colored_img = (colored_img[:, :, :3] * 255).astype(np.uint8)
        else:
            colored_img = np.stack([norm_img]*3, axis=-1)

        # Escalar si se desea
        if scale > 1:
            colored_img = colored_img.repeat(scale, axis=0).repeat(scale, axis=1)

        # Guardar como PNG
        filename = save_dir / f"{prefix}_{i:03d}.png"
        imageio.imwrite(filename, colored_img)

    if verbose:
        print(f"Saved PNG Images")

def plot_quantity_vs_T(Ts, values, errors=None, ylabel="", title="", save_path=None, color='b', marker='o', connect_points=True):
    """
    Plot magnitudes térmicas vs temperatura con estilo de artículo científico.
    - Añade barras de error si errors ≠ None.
    - Permite puntos más discretos para que no oculten errores pequeños.
    """
    plt.figure(figsize=(6, 4))

    if errors is not None:
        plt.errorbar(
            Ts, values, yerr=errors,
            fmt=marker,
            color=color,
            markersize=2.5,        # más pequeño
            elinewidth=1.2,        # más grueso
            capsize=3,             # barras más visibles
            linestyle='-' if connect_points else 'none',
            alpha=0.9              # opcional: punto semitransparente
        )
    else:
        plt.plot(
            Ts, values,
            marker=marker,
            markersize=2.5,
            linestyle='-' if connect_points else 'none',
            color=color,
            alpha=0.9
        )

    plt.xlabel('Temperature (T)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close()



def find_Tc(Ts: np.ndarray, heat_capacities: np.ndarray, std_Cv: np.ndarray) -> tuple[float, float]:
    """
    Estima Tc como el vértice de una parábola ajustada a C_v(T),
    ponderando por el error en C_v. Calcula también el error en Tc
    por propagación del ajuste.

    Parámetros:
    - Ts : array de temperaturas
    - heat_capacities : array de C_v
    - std_Cv : array de errores en C_v (desviaciones estándar)

    Retorna:
    - Tc : estimación del punto crítico
    - sigma_Tc : error estimado de Tc
    """
    window_size = 7  # número impar de puntos alrededor del máximo
    idx_peak = np.argmax(heat_capacities)
    half_w = window_size // 2

    i_start = max(0, idx_peak - half_w)
    i_end = min(len(Ts), idx_peak + half_w + 1)

    if (i_end - i_start) < 3:
        return Ts[idx_peak], np.nan

    x = Ts[i_start:i_end]
    y = heat_capacities[i_start:i_end]
    sigma_y = std_Cv[i_start:i_end]

    weights = 1.0 / (sigma_y**2 + 1e-12)  # evitar división por cero

    coeffs, cov = np.polyfit(x, y, 2, w=weights, cov=True)
    a, b, _ = coeffs

    Tc = -b / (2 * a)

    # Propagación del error
    var_a = cov[0, 0]
    var_b = cov[1, 1]
    cov_ab = cov[0, 1]

    dTc_da = b / (2 * a**2)
    dTc_db = -1 / (2 * a)

    sigma_Tc = np.sqrt(
        (dTc_da ** 2) * var_a +
        (dTc_db ** 2) * var_b +
        2 * dTc_da * dTc_db * cov_ab
    )

    return Tc, sigma_Tc




def extrapolate_Tc(N_values: np.ndarray, Tc_values: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    Igual que extrapolate_exponent pero para Tc
    """
    invN = 1.0 / N_values
    coef, cov = np.polyfit(invN, Tc_values, 1, cov=True)
    Tc_inf = coef[1]
    err = np.sqrt(cov[1, 1])
    return Tc_inf, err, coef




def get_clustered_temperatures(n_temperatures, center, low, high, fraction_center=0.7, width=0.15):
    """
    Genera un conjunto de temperaturas deterministas, distribuidas densamente
    alrededor de un valor central. Se garantiza que una fracción configurable de los
    puntos quede dentro de un intervalo simétrico alrededor del centro.

    Parámetros:
    -----------
    - n_temperatures : int
        Número total de temperaturas.
    - center : float
        Temperatura alrededor de la cual se desea mayor densidad.
    - low : float
        Límite inferior del rango total.
    - high : float
        Límite superior del rango total.
    - fraction_center : float
        Fracción de los puntos que deben estar en [center - width, center + width].
    - width : float
        Mitad del intervalo denso alrededor del centro.

    Retorna:
    --------
    - numpy.ndarray
        Array de temperaturas ordenadas y más densas alrededor de 'center'.
    """
    import numpy as np

    # 1. Calcular cuántos puntos van al centro
    n_center = int(n_temperatures * fraction_center)
    n_side = (n_temperatures - n_center) // 2
    remainder = n_temperatures - (2 * n_side + n_center)  # por si es impar

    # 2. Partes: izquierda (sparse), centro (dense), derecha (sparse)
    Ts_left = np.linspace(low, center - width, n_side, endpoint=False)
    Ts_center = np.linspace(center - width, center + width, n_center, endpoint=False)
    Ts_right = np.linspace(center + width, high, n_side + remainder, endpoint=True)

    # 3. Unir y retornar
    Ts = np.concatenate([Ts_left, Ts_center, Ts_right])
    return np.sort(Ts)



def estimate_critical_exponents(
    Ts, mags, std_mags,
    heat_capacities, std_Cv,
    susceptibilities, std_chi,
    Tc, window=0.16, min_points=5,
    min_dist_to_Tc=0.04  # Nuevo parámetro para excluir puntos demasiado cercanos a Tc
):
    """
    Estima β, α y γ con errores estándar usando ajuste log-log ponderado por las incertidumbres.

    Parámetros:
    - Ts : array de temperaturas
    - mags : array de magnetizaciones medias
    - std_mags : array de errores estándar de las magnetizaciones
    - heat_capacities : array de calor específico medio
    - std_Cv : array de errores estándar del calor específico
    - susceptibilities : array de susceptibilidades
    - std_chi : array de errores estándar de susceptibilidad
    - Tc : temperatura crítica estimada
    - window : ancho del intervalo alrededor de Tc para el ajuste
    - min_points : número mínimo de puntos para el ajuste
    - min_dist_to_Tc : distancia mínima a Tc para excluir puntos demasiado cercanos

    Retorna:
    - beta_fit : exponente β
    - beta_err : error en β
    - alpha_fit : exponente α
    - alpha_err : error en α
    - gamma_fit : exponente γ
    - gamma_err : error en γ
    - mask_critical : máscara booleana de puntos usados en el ajuste
    """

    delta_T = np.abs(Ts - Tc)
    # Nueva máscara que excluye puntos demasiado cercanos a Tc
    mask_critical = (delta_T < window) & (delta_T > min_dist_to_Tc) & (Ts < Tc)

    if np.count_nonzero(mask_critical) < min_points:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, mask_critical

    # Subconjuntos críticos
    T_vals = Ts[mask_critical]
    M_vals = mags[mask_critical]
    C_vals = heat_capacities[mask_critical]
    chi_vals = susceptibilities[mask_critical]

    log_T = np.log(Tc - T_vals)
    log_M = np.log(M_vals + 1e-10)
    log_C = np.log(C_vals + 1e-10)
    log_chi = np.log(chi_vals + 1e-10)

    # Errores relativos
    log_M_errors = std_mags[mask_critical] / (M_vals + 1e-10)
    log_C_errors = 2*std_Cv[mask_critical] / (C_vals + 1e-10)
    log_chi_errors = std_chi[mask_critical] / (chi_vals + 1e-10)

    # Pesos para ajuste ponderado
    weights_M = 1.0 / (log_M_errors**2 + 1e-12)
    weights_C = 1.0 / (log_C_errors**2 + 1e-12)
    weights_chi = 1.0 / (log_chi_errors**2 + 1e-12)

    # Ajustes lineales log-log
    slope_M, _ = np.polyfit(log_T, log_M, 1, w=weights_M)
    slope_C, _ = np.polyfit(log_T, log_C, 1, w=weights_C)
    slope_chi, _ = np.polyfit(log_T, log_chi, 1, w=weights_chi)

    beta_fit = slope_M
    alpha_fit = -slope_C
    gamma_fit = -slope_chi

    # Estimación de errores a partir de la matriz de covarianza
    try:
        _, cov_beta = np.polyfit(log_T, log_M, 1, w=weights_M, cov=True)
        beta_err = np.sqrt(cov_beta[0, 0])
    except:
        beta_err = np.nan

    try:
        _, cov_alpha = np.polyfit(log_T, log_C, 1, w=weights_C, cov=True)
        alpha_err = np.sqrt(cov_alpha[0, 0])
    except:
        alpha_err = np.nan

    try:
        _, cov_gamma = np.polyfit(log_T, log_chi, 1, w=weights_chi, cov=True)
        gamma_err = np.sqrt(cov_gamma[0, 0])
    except:
        gamma_err = np.nan

    return beta_fit, beta_err, alpha_fit, alpha_err, gamma_fit, gamma_err, mask_critical

def extrapolate_exponent(N_values: np.ndarray, exponent_values: np.ndarray, exponent_errors: np.ndarray, label='β') -> tuple[float, float, np.ndarray]:
    """
    Extrapola el exponente al límite N→∞ con ajuste ponderado por los errores (1σ).
    
    Devuelve:
    - valor extrapolado en N→∞
    - error (1σ)
    - coeficientes del ajuste lineal (pendiente, ordenada)
    """
    invN = 1.0 / N_values
    weights = 1.0 / (exponent_errors**2 + 1e-12)  # Evitar división por cero

    coef, cov = np.polyfit(invN, exponent_values, 1, w=weights, cov=True)
    exponent_inf = coef[1]
    err = np.sqrt(cov[1, 1])

    print(f"Estimación {label}(∞) = {exponent_inf:.3f} ± {err:.3f} (pendiente={coef[0]:.3f})")
    return exponent_inf, err, coef
