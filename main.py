# Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import random
from pathlib import Path
import timeit

from Ising_Functions import *

# paths

test_mode = False

if test_mode:
    print("Test mode activado")
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)
    figures_dir = Path("test") / "figures"
    figures_dir.mkdir(exist_ok=True)
    data_dir = Path("test") / "data"
    data_dir.mkdir(exist_ok=True)
    gifs_dir = Path("test") / "gifs"
    gifs_dir.mkdir(exist_ok=True)

else:
    # Paths para guardar imágenes y resultados
    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)  # Crear directorio si no existe

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)  # Crear directorio si no existe

    gifs_dir = Path("gifs")
    gifs_dir.mkdir(exist_ok=True)  # Crear directorio si no existe


# Variables de control


# Configuración de la simulación
N = 128                # Tamaño de la red
T = 0.75              # Temperatura (en unidades arbitrarias)
pasos = 10000           # Número de pasos Monte Carlo
num_imagenes = 5      # Número de imágenes a guardar para el paper

# Reproducibilidad
np.random.seed(42)
random.seed(42)

if test_mode:
    N = 128
    T = 0.75
    pasos = 250
    num_imagenes = 5

# Variables de control comunes
generar_gif = True
intervalo_gif = 100
calcular_magnetizacion = True
intervalo_magnetizacion = 100
verbose = False

# Opción de modelo: "clasico", "modificado" o "ambos"
modelo = "ambos"  # Cambia este valor segun la simulación deseada



if modelo == "modificado" or modelo == "ambos":
    if verbose:
        print(f"Simulando Modelo de Ising modificado (con interacciones de next-neighbours) para T = {T}, N = {N}...")
    frames_mod, mags_mod, pasos_MC_frames_mod, pasos_MC_mags_mod = ising_simulacion_modificado(
        N, T, pasos=pasos,
        generar_gif=generar_gif, intervalo_gif=intervalo_gif,
        calcular_magnetizacion=calcular_magnetizacion, intervalo_magnetizacion=intervalo_magnetizacion
    )

    if generar_gif:
        gif_name_mod = f"ising_mod_T{T}_N{N}.gif"
        if verbose:
            print(f"Guardando animación del modelo modificado como {gif_name_mod}...")
        crear_gif(frames_mod, nombre=gif_name_mod, dir_path=gifs_dir)

    if calcular_magnetizacion:
        mag_filepath_mod = data_dir / f"magnetizacion_mod_T{T}_N{N}.txt"
        if verbose:
            print(f"Guardando magnetización promedio (modelo modificado) en {mag_filepath_mod}...")
        np.savetxt(mag_filepath_mod, mags_mod)
        if verbose:
            print("Magnetización (modificado) guardada correctamente.")

        curva_magnetizacion_mod = f"magnetizacion_curva_mod_T{T}_N{N}.png"
        graficar_magnetizacion(mags_mod, pasos_MC_mags_mod, T, N, nombre=curva_magnetizacion_mod, dir_path=figures_dir)
        if verbose:
            print(f"Curva de magnetización (modificado) guardada como {curva_magnetizacion_mod}.")

    if verbose:
        print("Guardando imágenes equiespaciadas (modelo modificado) para el paper...")
    guardar_imagenes(frames_mod, num_imagenes=num_imagenes, nombre_base=f"ising_mod_T{T}_N{N}", dir_path=figures_dir, verbose=verbose)



if modelo == "clasico" or modelo == "ambos":
    if verbose:
        print(f"\nSimulando Modelo de Ising clásico para T = {T}, N = {N}...")
    frames_cls, mags_cls, pasos_MC_frames_cls, pasos_MC_mags_cls = ising_simulacion_clasico(
        N, T, pasos=pasos,
        generar_gif=generar_gif, intervalo_gif=intervalo_gif,
        calcular_magnetizacion=calcular_magnetizacion, intervalo_magnetizacion=intervalo_magnetizacion
    )

    if generar_gif:
        gif_name_cls = f"ising_cls_T{T}_N{N}.gif"
        if verbose:
            print(f"Guardando animación del modelo clásico como {gif_name_cls}...")
        crear_gif(frames_cls, nombre=gif_name_cls, dir_path=gifs_dir)

    if calcular_magnetizacion:
        mag_filepath_cls = data_dir / f"magnetizacion_cls_T{T}_N{N}.txt"
        if verbose:
            print(f"Guardando magnetización promedio (modelo clásico) en {mag_filepath_cls}...")
        np.savetxt(mag_filepath_cls, mags_cls)
        if verbose:
            print("Magnetización (clásico) guardada correctamente.")

        curva_magnetizacion_cls = f"magnetizacion_curva_cls_T{T}_N{N}.png"
        graficar_magnetizacion(mags_cls, pasos_MC_mags_cls, T, N, nombre=curva_magnetizacion_cls, dir_path=figures_dir)
        if verbose:
            print(f"Curva de magnetización (clásico) guardada como {curva_magnetizacion_cls}.")

    if verbose:
        print("Guardando imágenes equiespaciadas (modelo clásico) para el paper...")
    guardar_imagenes(frames_cls, num_imagenes=num_imagenes, nombre_base=f"ising_cls_T{T}_N{N}", dir_path=figures_dir, verbose=verbose)
if verbose:
    print("\nSimulación completada. Resultados guardados correctamente.")