import numpy as np
import matplotlib
matplotlib.use("Agg")  # Usar backend sin GUI para guardar imágenes/ GIFs
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
import random

custom_cmap = LinearSegmentedColormap.from_list("red_blue", ["#0033CC", "#CC0000"])

# Constantes para interacciones (nuevos valores)
J1 = 1.0   # Acoplamiento para vecinos inmediatos
J2 = 2.0   # Acoplamiento para vecinos diagonales (next-nearest neighbors)

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------

def puntoaleatorio(N):
    """
    Devuelve una coordenada aleatoria válida dentro de la red de NxN.
    (excluyendo los bordes de la matriz extendida N+2xN+2)
    """
    return random.randint(1, N), random.randint(1, N)

def P(s, n, m, T):
    """
    Calcula la probabilidad de aceptar un cambio de spin en (n, m) a temperatura T.
    Basado en el criterio de Metropolis, considerando vecinos inmediatos y
    vecinos diagonales (next-nearest neighbors).
    """
    # Suma de vecinos inmediatos
    vecinos_inmediatos = s[n+1, m] + s[n-1, m] + s[n, m+1] + s[n, m-1]
    # Suma de vecinos diagonales
    vecinos_diagonales = s[n+1, m+1] + s[n-1, m+1] + s[n+1, m-1] + s[n-1, m-1]

    # Cálculo del cambio de energía: ΔE = 2 * s(n, m) * (J1*(vecinos inmediatos) + J2*(vecinos diagonales))
    E = 2 * s[n, m] * (J1 * vecinos_inmediatos + J2 * vecinos_diagonales)
    return min(1.0, np.exp(-E / T)) 

def est_prob(p, s, n, m):
    """
    Devuelve el nuevo valor del spin en (n, m) dependiendo de p.
    """
    return -s[n, m] if np.random.random() < p else s[n, m]

def crear_matriz(N):
    """
    Crea una red de NxN spins aleatorios (valores +1 o -1) y añade bordes periódicos.
    """
    red = np.where(np.random.rand(N, N) < 0.5, -1, 1)
    web = np.zeros((N + 2, N + 2))
    web[1:N + 1, 1:N + 1] = red
    web[0, 1:N + 1] = red[N - 1, :]
    web[N + 1, 1:N + 1] = red[0, :]
    web[1:N + 1, 0] = red[:, N - 1]
    web[1:N + 1, N + 1] = red[:, 0]
    # Inicializar las esquinas
    web[0,0] = red[N-1, N-1]
    web[0, N+1] = red[N-1, 0]
    web[N+1, 0] = red[0, N-1]
    web[N+1, N+1] = red[0, 0]
    return web

def actualizar_contornos(web, N):
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

# -----------------------------
# SIMULACIÓN PRINCIPAL
# -----------------------------

def ising_simulacion(N, T, pasos=100, generar_gif=True, calcular_magnetizacion=False):
    """
    Ejecuta la simulación del modelo de Ising modificado con interacciones de corto y largo alcance:
    - N: tamaño de la red NxN
    - T: temperatura del sistema
    - pasos: número de pasos Monte Carlo
    - generar_gif: si True, guarda los estados intermedios para animación
    - calcular_magnetizacion: si True, calcula la magnetización en cada paso

    Se guarda la imagen del estado inicial y luego los pasos se van grabando.
    """
    s = crear_matriz(N)
    frames = []
    magnetizaciones = []

    # Guardar el estado inicial
    if generar_gif:
        frames.append(s[1:N + 1, 1:N + 1].copy())
    if calcular_magnetizacion:
        mag = np.abs(np.sum(s[1:N + 1, 1:N + 1])) / (N * N)
        magnetizaciones.append(mag)

    for paso in range(pasos):
        for _ in range(N * N):
            x, y = puntoaleatorio(N)
            p_val = P(s, x, y, T)
            nuevo = est_prob(p_val, s, x, y)
            if s[x, y] != nuevo:
                s[x, y] = nuevo
                s = actualizar_contornos(s, N)

        if generar_gif:
            frames.append(s[1:N + 1, 1:N + 1].copy())

        if calcular_magnetizacion:
            mag = np.abs(np.sum(s[1:N + 1, 1:N + 1])) / (N * N)
            magnetizaciones.append(mag)

    return frames, magnetizaciones

# -----------------------------
# ANIMACIÓN
# -----------------------------

def crear_gif(frames, nombre="ising.gif", intervalo=100, dpi=150):
    """
    Crea un archivo .gif animado desde la lista de fotogramas (matrices de spin).
    """
    fig, ax = plt.subplots()
    ax.axis("off")
    im = ax.imshow(frames[0], cmap=custom_cmap, vmin=-1, vmax=1)

    def update(i):
        im.set_data(frames[i])
        return im,

    ani = FuncAnimation(fig, update, frames=len(frames), interval=intervalo, blit=True)
    ani.save(nombre, dpi=dpi, writer=PillowWriter(fps=1000//intervalo))
    plt.close()

# -----------------------------
# GUARDAR IMÁGENES EQUIESPACIADAS
# -----------------------------

def guardar_imagenes(frames, num_imagenes=5, nombre_base="ising", dpi=150):
    """
    Guarda num_imagenes imágenes del sistema de Ising a partir de la lista de frames.
    Se seleccionan imágenes equiespaciadas, incluyendo la primera y la última.

    Parámetros:
    - frames: lista de matrices de spin.
    - num_imagenes: número de imágenes a guardar (por defecto 5).
    - nombre_base: base para el nombre de archivo de las imágenes.
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
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"Imagen guardada: {filename}")

# -----------------------------
# REPRESENTACIÓN DE LA CURVA DE MAGNETIZACIÓN
# -----------------------------

def graficar_magnetizacion(magnetizaciones, T, N, nombre="magnetizacion_curva.png"):
    """
    Crea y guarda un gráfico de la magnetización en función del número de pasos Monte Carlo.

    Parámetros:
    - magnetizaciones: lista de valores de magnetización obtenidos en la simulación.
    - T: temperatura del sistema (para etiquetar el gráfico).
    - N: tamaño de la red (para etiquetar el gráfico).
    - nombre: nombre del archivo de salida (imagen PNG).
    """
    plt.figure()
    plt.plot(magnetizaciones, marker='o', linestyle='-', color='blue')
    plt.xlabel("Pasos Monte Carlo")
    plt.ylabel("Magnetización promedio")
    plt.title(f"Evolución de la Magnetización - T = {T}, N = {N}")
    plt.grid(False)
    plt.savefig(nombre, dpi=150)
    plt.close()

# -----------------------------
# EJECUCIÓN PRINCIPAL
# -----------------------------

if __name__ == "__main__":
    N = 64                # Tamaño de la red
    T = 0.5               # Temperatura (en unidades arbitrarias)
    pasos = 100          # Número de pasos Monte Carlo
    gif_name = f"ising_T{T}_N{N}.gif"
    num_imagenes = 5      # Número de imágenes a guardar para el paper

    print(f"Simulando Modelo de Ising modificado (con interacciones de largo alcance) para T = {T}, N = {N}...")
    frames, mags = ising_simulacion(N, T, pasos=pasos, generar_gif=True, calcular_magnetizacion=True)

    print(f"Guardando animación como {gif_name}...")
    crear_gif(frames, nombre=gif_name)

    np.savetxt(f"magnetizacion_T{T}_N{N}.txt", mags)
    print("Simulación completada. Resultados guardados correctamente.")

    # Representar la curva de magnetización
    curva_magnetizacion = f"magnetizacion_curva_T{T}_N{N}.png"
    graficar_magnetizacion(mags, T, N, nombre=curva_magnetizacion)
    print(f"Curva de magnetización guardada como {curva_magnetizacion}.")

    # Guardar imágenes equiespaciadas para presentación en paper científico
    print("Guardando imágenes equiespaciadas para el paper...")
    guardar_imagenes(frames, num_imagenes=num_imagenes, nombre_base=f"ising_T{T}_N{N}")

