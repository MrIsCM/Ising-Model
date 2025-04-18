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

def get_energy(lattice, N, J1, J2):
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
        The total energy of the lattice normalized by the number of spins (N*N).
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
def compute_specific_heat(energy_array, N, T):
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
    
    C = np.var(energy_array) / (T * N*N)

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
def metropolis(lattice, MC_steps, T, energy, N, J1, J2, save_images=False, image_spacing=None, verbose=0):
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
    - image_spacing : list of int, optional
        List of Monte Carlo steps at which to save lattice snapshots (default is None).
    - verbose : int, optional
        Verbosity level for logging information during the simulation:
        - 0: No output.
        - 1: Basic information.
        - 2: Detailed information (default is 0).
    Returns:
    --------
    - net_spins : numpy.ndarray
        Array of net magnetization values at each Monte Carlo step.
    - net_energy : numpy.ndarray
        Array of energy values at each Monte Carlo step.
    - images : numpy.ndarray or None
        Array of saved lattice snapshots if `save_images` is True, otherwise None.
    - web[-1] : numpy.ndarray
        Final lattice configuration after the simulation.
    Notes:
    ------
    - The Metropolis algorithm is used to simulate the evolution of the Ising model.
    - The flipping condition is determined by the change in energy (dE) and the temperature (T).
    - If `save_images` is True, the lattice snapshots are saved at the specified `image_spacing` steps.
    - If the system is not in equilibrium after the simulation, it is possible to run again using the returned last lattice configuration as the new initial state. 
    """


    # 1. Initialize variables
    web = lattice.copy()
    net_spins = np.zeros(MC_steps)
    net_energy = np.zeros(MC_steps)

    # Information output. Ammount if information printed is controlled by verbose.
    if verbose > 0:
        print(f"Starting simulation with configuration:")
        print(f"\t - NxN: {N}x{N}")
        print(f"\t - Temperature: {T}")
        print(f"\t - J1: {J1}")
        print(f"\t - J2: {J2}")
        print(f"\t - MC steps: {MC_steps}")
    if verbose > 1:
        print(f"\t - Initial energy: {energy}")
        print(f"\t - Initial magnetization: {web.sum()/(N**2)}")

    #------------------------
    #   Image saving logic
    #------------------------
    if save_images and image_spacing is not None:
        if verbose > 0:
            print(f"This run wil save {len(image_spacing)} images")
        images = np.empty((len(image_spacing), N, N), dtype=np.int8)
        aux_img_idx = 0
    # 'None' used for consistency in the return statement
    else:
        if verbose > 0:
            print(f"This run will not save images")
        images = None


    # ---------------------
    #       Main loop
    # ---------------------
    for t in range(MC_steps):
        if save_images and t in image_spacing:
            if verbose > 1:
                print(f"Saving state at step: {t}/{MC_steps}")
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

        last_config = web.copy()

    return net_spins, net_energy, images, last_config

def path_configuration(N, T, J1=None, J2=None, data_dir='data', figures_dir='figures', images_dir='images', verbose=0):
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
    
    # Parent folder
    if verbose > 1:
        print(f"Creating parent folder")
    if J1 is None or J2 is None:
        parent_name = f"Simulation_N{N}_T{T}"
    else:
        parent_name = f"Simulation_N{N}_T{T}_J1{J1}_J2{J2}"
    parent_dir = Path(parent_name)
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