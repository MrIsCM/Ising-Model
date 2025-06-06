# Imports
import numpy as np
import matplotlib.pyplot as plt
import numba
from numba import njit
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
    C = np.var(energy_array[burn_in_index:]) / (T * N*N)

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


@njit(parallel=True)
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
    

@njit(parallel=True)
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