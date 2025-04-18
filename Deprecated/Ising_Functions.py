# Imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import random
from pathlib import Path

# Parametros/configuración de las librerías
matplotlib.use("Agg")  # Usar backend sin GUI para guardar imágenes/ GIFs
custom_cmap = LinearSegmentedColormap.from_list("red_blue", ["#0033CC", "#CC0000"])


# Constantes para interacciones (nuevos valores para el modelo modificado)
J1 = 1.0   # Acoplamiento para vecinos inmediatos
J2 = 2.0   # Acoplamiento para vecinos diagonales (next-nearest neighbors)

# -----------------------------
# FUNCIONES AUXILIARES COMUNES
# -----------------------------

def puntoaleatorio(N):
    """
    Devuelve una coordenada aleatoria válida dentro de la red de NxN.
    (excluyendo los bordes de la matriz extendida N+2xN+2)
    """
    return random.randint(1, N), random.randint(1, N)

def est_prob(p, s, n, m):
    """
    Devuelve el nuevo valor del spin en (n, m) dependiendo de p.
    """
    return -s[n, m] if np.random.random() < p else s[n, m]

# -----------------------------
# FUNCIONES PARA EL MODELO MODIFICADO (con next-neighbours)
# -----------------------------

def P_modificado(s, n, m, T, J1=J1, J2=J2):
    """
    Calcula la probabilidad de aceptar un cambio de spin en (n, m) a temperatura T,
    considerando vecinos inmediatos y vecinos diagonales (next-nearest neighbors).
    Basado en el criterio de Metropolis.
    """
    # Suma de vecinos inmediatos
    vecinos_inmediatos = s[n+1, m] + s[n-1, m] + s[n, m+1] + s[n, m-1]
    # Suma de vecinos diagonales
    vecinos_diagonales = s[n+1, m+1] + s[n-1, m+1] + s[n+1, m-1] + s[n-1, m-1]

    # Cálculo del cambio de energía: ΔE = 2 * s(n, m) * (J1*(vecinos inmediatos) + J2*(vecinos diagonales))
    E = 2 * s[n, m] * (J1 * vecinos_inmediatos + J2 * vecinos_diagonales)
    return min(1.0, np.exp(-E / T))


def crear_matriz_toroidal(N, values=[1, -1], padding_width=1, mode='wrap', seed=42):
    """
    Crea una red de NxN spins aleatorios (valores +1 o -1) y añade bordes periódicos toroidales.
    
    Parámetros:
    - N (int): Tamaño de la red (NxN).
    - padding_width (int): Ancho del borde añadido (por defecto 1).
    - mode (str): Modo de relleno para los bordes (por defecto 'wrap' para bordes periódicos).

    Devuelve:
    - numpy.ndarray: Red con bordes periódicos añadidos.
    """

    rng_generator = np.random.default_rng(seed=seed)
    
    red = rng_generator.choice(values, size=(N, N))
    web = np.pad(red, pad_width=padding_width, mode=mode)

    return web

def actualizar_contornos_modificado(web, N):
    """
    Actualiza los bordes de la red para mantener condiciones periódicas,
    incluyendo las esquinas, necesarias para las interacciones diagonales.
    """
    # Bordes horizontales
    web[0, 1:N + 1] = web[N, 1:N + 1]
    web[N + 1, 1:N + 1] = web[1, 1:N + 1]

    # Bordes verticales
    web[1:N + 1, 0] = web[1:N + 1, N]
    web[1:N + 1, N + 1] = web[1:N + 1, 1]

    # Esquinas
    web[0,0] = web[N, N]
    web[0, N+1] = web[N, 1]
    web[N+1, 0] = web[1, N]
    web[N+1, N+1] = web[1, 1]
    return web

def ising_simulacion_modificado(N, T, pasos=100, generar_gif=True, intervalo_gif=1,
                                calcular_magnetizacion=False, intervalo_magnetizacion=1, seed=None):
    """
    Ejecuta la simulación del modelo de Ising modificado (con interacciones de corto y largo alcance):
    - N: tamaño de la red NxN
    - T: temperatura del sistema
    - pasos: número de pasos Monte Carlo
    - generar_gif: si True, guarda los estados intermedios para animación
    - intervalo_gif: intervalo entre frames en el gif (en pasos Monte Carlo)
    - calcular_magnetizacion: si True, calcula la magnetización en cada paso
    - intervalo_magnetizacion: intervalo para calcular la magnetización (en pasos Monte Carlo)

    Se guarda la imagen del estado inicial y luego los pasos se van grabando.
    **Actualiza los contornos cada paso MC.**

    Devuelve:
    - frames: lista de matrices de spin para cada frame del gif
    - magnetizaciones: lista de valores de magnetización promedio
    - pasos_MC_frames: pasos Monte Carlo correspondientes a los frames
    - pasos_MC_mag: pasos Monte Carlo correspondientes a las magnetizaciones
    """
    s = crear_matriz_toroidal(N, seed=seed)

    # Inicializa listas para guardar frames y magnetizaciones
    frames = []
    pasos_MC_frames = []

    magnetizaciones = []
    pasos_MC_mag = []

    for paso in range(pasos):
        # Guardar estado para animación y medición
        if generar_gif and (paso % intervalo_gif == 0):
            frames.append(s[1:N + 1, 1:N + 1].copy())
            pasos_MC_frames.append(paso)

        if calcular_magnetizacion and (paso % intervalo_magnetizacion == 0):
            mag = np.abs(np.sum(s[1:N + 1, 1:N + 1])) / (N * N)
            magnetizaciones.append(mag)
            pasos_MC_mag.append(paso)

        # Bucle interno: N^2 intentos de cambio de spin
        for _ in range(N * N):
            x, y = puntoaleatorio(N)
            p_val = P_modificado(s, x, y, T)
            nuevo = est_prob(p_val, s, x, y)
            if s[x, y] != nuevo:
                s[x, y] = nuevo
        # Actualización de contornos una vez por paso Monte Carlo
        s = actualizar_contornos_modificado(s, N)

    # Guardar el último estado si no se guardó previamente (con verificación de lista vacía)
    if generar_gif and (len(pasos_MC_frames) == 0 or pasos_MC_frames[-1] != pasos):
        frames.append(s[1:N + 1, 1:N + 1].copy())
        pasos_MC_frames.append(pasos)

    if calcular_magnetizacion and (len(pasos_MC_mag) == 0 or pasos_MC_mag[-1] != pasos):
        mag = np.abs(np.sum(s[1:N + 1, 1:N + 1])) / (N * N)
        magnetizaciones.append(mag)
        pasos_MC_mag.append(pasos)

    return frames, magnetizaciones, pasos_MC_frames, pasos_MC_mag


# -----------------------------
# FUNCIONES PARA EL MODELO CLÁSICO
# -----------------------------

def P_clasico(s, n, m, T):
    """
    Calcula la probabilidad de aceptar un cambio de spin en (n, m) a temperatura T,
    usando el criterio de Metropolis y considerando solo vecinos inmediatos.
    """
    E = 2 * s[n, m] * (s[n+1, m] + s[n-1, m] + s[n, m+1] + s[n, m-1])
    return min(1.0, np.exp(-E / T))



def actualizar_contornos_clasico(web, N):
    """
    Actualiza los bordes de la red para mantener condiciones periódicas, considerando
    solo la actualización de los límites (no se tocan las esquinas).
    """
    web[0, 1:N+1] = web[N, 1:N+1]
    web[N+1, 1:N+1] = web[1, 1:N+1]
    web[1:N+1, 0] = web[1:N+1, N]
    web[1:N+1, N+1] = web[1:N+1, 1]
    return web

def ising_simulacion_clasico(N, T, pasos=100, generar_gif=True, intervalo_gif=1,
                             calcular_magnetizacion=False, intervalo_magnetizacion=1, seed=None):
    """
    Ejecuta la simulación del modelo clásico de Ising:
    - N: tamaño de la red NxN
    - T: temperatura del sistema
    - pasos: número de pasos Monte Carlo
    - generar_gif: si True, guarda los estados intermedios para animación
    - intervalo_gif: intervalo entre frames en el gif (en pasos Monte Carlo)
    - calcular_magnetizacion: si True, calcula la magnetización en cada paso
    - intervalo_magnetizacion: intervalo para calcular la magnetización (en pasos Monte Carlo)

    IMPORTANTE: Se guarda la imagen del estado inicial y luego los pasos se van grabando.
    **Actualiza los contornos cada paso MC.**

    Devuelve:
    - frames: lista de matrices de spin para cada frame del gif
    - magnetizaciones: lista de valores de magnetización promedio
    - (para mantener la consistencia, se devuelven también los pasos MC correspondientes)
    """
    s = crear_matriz_toroidal(N, seed=seed)

    frames = []
    pasos_MC_frames = []
    magnetizaciones = []
    pasos_MC_mag = []

    for paso in range(pasos):
        if generar_gif and (paso % intervalo_gif == 0):
            frames.append(s[1:N+1, 1:N+1].copy())
            pasos_MC_frames.append(paso)
        if calcular_magnetizacion and (paso % intervalo_magnetizacion == 0):
            mag = np.abs(np.sum(s[1:N+1, 1:N+1]))/(N*N)
            magnetizaciones.append(mag)
            pasos_MC_mag.append(paso)
        for _ in range(N * N):
            x, y = puntoaleatorio(N)
            p_val = P_clasico(s, x, y, T)
            nuevo = est_prob(p_val, s, x, y)
            if s[x, y] != nuevo:
                s[x, y] = nuevo
        s = actualizar_contornos_clasico(s, N)

    if generar_gif and (len(pasos_MC_frames) == 0 or pasos_MC_frames[-1] != pasos):
        frames.append(s[1:N+1, 1:N+1].copy())
        pasos_MC_frames.append(pasos)
    if calcular_magnetizacion and (len(pasos_MC_mag) == 0 or pasos_MC_mag[-1] != pasos):
        mag = np.abs(np.sum(s[1:N+1, 1:N+1]))/(N*N)
        magnetizaciones.append(mag)
        pasos_MC_mag.append(pasos)

    return frames, magnetizaciones, pasos_MC_frames, pasos_MC_mag


# -----------------------------
# FUNCIONES DE VISUALIZACIÓN Y GUARDADO (COMUNES)
# -----------------------------

def crear_gif(frames, nombre="ising.gif", intervalo=100, dpi=150, dir_path=Path("gifs")):
    """
    Crea un archivo .gif animado desde la lista de fotogramas (matrices de spin).
    """
    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0], cmap=custom_cmap, vmin=-1, vmax=1)

    def update(i):
        im.set_data(frames[i])
        return im,

    file_path = dir_path / nombre
    ani = FuncAnimation(fig, update, frames=len(frames), interval=intervalo, blit=True)
    ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=1000//intervalo))
    plt.close(fig)

def guardar_imagenes(frames, num_imagenes=5, nombre_base="ising", dpi=150, dir_path=Path("figures"), verbose=True):
    """
    Guarda num_imagenes imágenes del sistema de Ising a partir de la lista de frames.
    Se seleccionan imágenes equiespaciadas, incluyendo la primera y la última.

    Parámetros:
    - frames: lista de matrices de spin.
    - num_imagenes: número de imágenes a guardar (por defecto 5).
    - nombre_base: base para el nombre de archivo de las imágenes.
    - dir_path: directorio donde se guardarán las imágenes.
    - verbose: si True, se imprime el progreso.
    """
    total = len(frames)
    if num_imagenes >= total:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, num=num_imagenes, dtype=int)

    for i, idx in enumerate(indices):
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(frames[idx], cmap=custom_cmap, vmin=-1, vmax=1)
        filename = f"{nombre_base}_imagen_{i+1}.png"
        file_path = dir_path / filename
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        if verbose:
            print(f"Imagen guardada: {filename}")

def graficar_magnetizacion(magnetizaciones, pasos_MC_mags, T, N, nombre="magnetizacion_curva.png", dir_path=Path("figures")):
    """
    Crea y guarda un gráfico de la magnetización en función del número de pasos Monte Carlo.

    Parámetros:
    - magnetizaciones: lista de valores de magnetización obtenidos en la simulación.
    - pasos_MC_mags: lista de pasos Monte Carlo correspondientes a las magnetizaciones.
    - T: temperatura del sistema (para etiquetar el gráfico).
    - N: tamaño de la red (para etiquetar el gráfico).
    - nombre: nombre del archivo de salida (imagen PNG).
    - dir_path: directorio donde se guardará la imagen.
    """
    plt.figure()
    plt.plot(pasos_MC_mags, magnetizaciones, marker='o', linestyle='--', color='blue')
    plt.xlabel("Pasos Monte Carlo")
    plt.ylabel("Magnetización promedio")
    plt.title(f"Evolución de la Magnetización - T = {T}, N = {N}")
    plt.grid(False)
    file_path = dir_path / nombre
    plt.savefig(file_path, dpi=150)
    plt.close()