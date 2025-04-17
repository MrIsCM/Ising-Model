import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from scipy.ndimage import convolve

import imageio
from PIL import Image

class IsingModel:

    def __init__(self, N, T, J1, J2, test_mode=False, seed=None):
        """
        Inicializa una instancia del modelo de Ising.
        ----------------------------------------------
        Parámetros:
        - N (int): El tamaño de la red (una cuadrícula de NxN).
        - T (float): La temperatura del sistema.
        - J1 (float): La fuerza de interacción entre vecinos más cercanos.
        - J2 (float): La fuerza de interacción entre los siguientes vecinos más cercanos.
        - SEED (int, opcional): La semilla para la generación de números aleatorios. Por defecto es None.
        """

        # Parámetros del modelo de Ising
        self.N = N
        self.NN = N * N
        self.T = T
        self.J1 = J1
        self.J2 = J2

        # Variables de control de la simulación
        self.test_mode = test_mode
        self.SEED = seed

        self.kernel_NN = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]])
        
        self.kernel_NNN = np.array([
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]])


    def inicia_red(self, valores=[1, -1], p=0.5):
        """
        Inicializa la red (lattice) para la simulación del modelo de Ising.
        - Parámetros:
            - valores (list, opcional): Una lista de posibles valores de espín para la red. 
                          Por defecto es [1, -1].
            - p (float, opcional): La probabilidad de que un espín tome el valor 1. Por defecto es 0.5.
        - Atributos:
            - web (numpy.ndarray): La red utilizada para la simulación.
        """
        rng = np.random.default_rng(self.SEED)
        self.web = rng.choice(valores, size=(self.N, self.N), p=[p, 1-p])
        self.web = self.web.astype(np.int8)  # Asegura que los valores son enteros de 8 bits

    def calcular_probabilidad(self, i, j):
        """
        Calcula la probabilidad de que un espín en la posición (i, j) cambie de estado.
        Parámetros:
            i (int): La fila del espín en la red.
            j (int): La columna del espín en la red.
        Devuelve:
            float: La probabilidad de que el espín cambie de estado.
        """
        # Extrae la red de espines
        s = self.web
        N = self.N

        # Calcula los índices de los vecinos inmediatos y diagonales
        # usando algebra modular para el contorno envolvente
        vecinos_inmediatos = (
            s[(i+1)%N, j] + s[(i-1)%N, j] +
            s[i, (j+1)%N] + s[i, (j-1)%N]
        )
        vecinos_diagonales = (
            s[(i+1)%N, (j+1)%N] + s[(i-1)%N, (j+1)%N] +
            s[(i+1)%N, (j-1)%N] + s[(i-1)%N, (j-1)%N]
        )

        # Calculo del cambio de energía
        dE = 2 * s[i, j] * (self.J1 * vecinos_inmediatos + self.J2 * vecinos_diagonales)

        return min(1, np.exp(-dE / self.T))
    
    def calcular_dE_precomp(self, i, j):
        
        S_nn = 0
        S_nnn = 0

        for vecino in self.vecinos_nn[(i,j)]:
            S_nn += self.web[vecino]
        
        for vecino in self.vecinos_nnn[(i,j)]:
            S_nnn += self.web[vecino]

        dE = 2 * (self.J1 * S_nn + self.J2 * S_nnn)

        return dE

    def calcular_energia(self):
        """
        Calcula la energía total del sistema basado en las interacciones 
        de los vecinos más cercanos y los vecinos de siguiente nivel.
        La energía se calcula utilizando convoluciones con dos kernels (Nearest Neighbor [nn] y Next Nearest Neighbor [nnn]):
        - `kernel_NN`: Representa las interacciones con los vecinos más cercanos.
        - `kernel_NNN`: Representa las interacciones con los vecinos de siguiente nivel.
        La fórmula para la energía total es:
            E = -J1 * Σ(web * energy_close_neigh) - J2 * Σ(web * energy_next_neigh)
        donde:
            - `web` es la configuración actual del sistema.
            - `J1` y `J2` son constantes que representan la intensidad de las interacciones con los vecinos más cercanos y los vecinos de siguiente nivel, respectivamente.
        Returns:
            - float: Energía total del sistema, dividida entre 2 para evitar contar las interacciones dos veces.
        """

        
        # Convolucion para obtener la energia total
        energy_nn = convolve(self.web, self.kernel_NN, mode='wrap')
        energy_nnn = convolve(self.web, self.kernel_NNN, mode='wrap')

        # Energia total
        energy = -self.J1 * np.sum(self.web * energy_nn) - self.J2 * np.sum(self.web * energy_nnn)

        return energy / 2  # Divido entre 2 porque cada interaccion se cuenta dos veces
    
    def precomputar_vecinos_cercanos(self):
        """
        Precomputa los vecinos más cercanos para cada espín en la red.
        ---------------------------------------------------------------
        Esto es más eficiente para simulaciones largas (muchos pasos Monte Carlo).
        """
        N = self.N
        vecinos_cercanos = {}
        for i in range(N):
            for j in range(N):
                vecinos_cercanos[(i, j)] = [
                    ((i+1)%N, j),
                    ((i-1)%N, j),
                    (i, (j+1)%N),
                    (i, (j-1)%N)
                ]
        self.vecinos_nn = vecinos_cercanos

    def precomputar_vecinos_diagonales(self):
        """
        Precomputa los vecinos de siguiente nivel para cada espín en la red.
        ---------------------------------------------------------------
        Esto es más eficiente para simulaciones largas (muchos pasos Monte Carlo).
        """
        N = self.N
        vecinos_diagonales = {}
        for i in range(N):
            for j in range(N):
                vecinos_diagonales[(i, j)] = [
                    ((i+1)%N, (j+1)%N),
                    ((i+1)%N, (j-1)%N),
                    ((i-1)%N, (j+1)%N),
                    ((i-1)%N, (j-1)%N)
                ]
        self.vecinos_nnn = vecinos_diagonales

    def precomputar_exp_dE(self):
        """
        Precomputa la exponencial de la variación de energía para cada espín en la red.
        ---------------------------------------------------------------
        Esto es más eficiente para simulaciones largas (muchos pasos Monte Carlo).
        """
        N = self.N
        exp_dE = {}
        possible_sums = [-4, -2, 0, 2, 4]
        for S_nn in possible_sums:
            for S_nnn in possible_sums:
                dE = 2 * (self.J1 * S_nn + self.J2 * S_nnn)
                exp_dE[dE] = np.exp(-dE / self.T)

        self.exp_dE = exp_dE

    def compute_dE_all(self):
        dE = (
            2 * self.web
            * (
                self.J1 * (
                    np.roll(self.web, 1, axis=0)
                    + np.roll(self.web, -1, axis=0)
                    + np.roll(self.web, 1, axis=1)
                    + np.roll(self.web, -1, axis=1)
                )
                + self.J2 * (
                    np.roll(self.web, (1, 1), axis=(0, 1))
                    + np.roll(self.web, (1, -1), axis=(0, 1))
                    + np.roll(self.web, (-1, 1), axis=(0, 1))
                    + np.roll(self.web, (-1, -1), axis=(0, 1))
                )
            )
        )

        return dE

    def simular(self, pasos_MC=1000, generar_gif=False, pasos_muestreo_frames=100, generar_imagenes=True, n_imagenes=5, espaciado_imagenes='log', calcular_magnetizacion=True, pasos_muestreo_magnetizacion=100, calcular_energia=False, pasos_muestreo_energia=10):

        # Generador de números aleatorios
        rng = np.random.default_rng(seed=self.SEED)

        # Pre-aloca espacio para los resultados en numpy arrays (mas eficiente)
        # Si no se quiere guardar algun resultado, se inicializan sus arrays como None
        # esto es necesario por consistencia en el return

        # Arrays para los Gifs
        if generar_gif:
            n_frames = pasos_MC // pasos_muestreo_frames + 1
            self.frames = np.empty((n_frames, self.N, self.N), dtype=np.int8)
            self.pasos_frames = np.empty(n_frames, dtype=int)
        else:
            self.frames = None
            self.pasos_frames = None

        # Arrays para las magnetizaciones
        if calcular_magnetizacion:
            n_magnetizaciones = pasos_MC // pasos_muestreo_magnetizacion + 1
            self.magnetizaciones = np.empty(n_magnetizaciones, dtype=float)
            self.pasos_magnetizaciones = np.empty(n_magnetizaciones, dtype=int)
        else:
            self.magnetizaciones = None
            self.pasos_magnetizaciones = None

        # Energia
        if calcular_energia:
            n_energia = pasos_MC // pasos_muestreo_energia + 1
            self.energia = np.empty(n_energia, dtype=float)
            self.pasos_energia = np.empty(n_energia, dtype=int)
        else:
            self.energia = None
            self.pasos_energia = None

        # Arrays para las imagenes
        if generar_imagenes:
            self.imagenes = np.empty((n_imagenes, self.N, self.N), dtype=np.int8)
            self.pasos_imagenes = np.empty(n_imagenes, dtype=int)
            if espaciado_imagenes == 'lineal':
                pasos_imagenes = np.linspace(0, pasos_MC, n_imagenes, dtype=int)
            elif espaciado_imagenes == 'log':
                pasos_imagenes = np.logspace(0, np.log10(pasos_MC), n_imagenes, dtype=int)
            aux_idx = 0
        else:
            self.imagenes = None
            self.pasos_imagenes = None


        # Simulacion
        for paso in range(pasos_MC):
            # Guarda el estado inicial
            if generar_gif and paso % pasos_muestreo_frames == 0:
                self.frames[paso // pasos_muestreo_frames] = self.web.copy()
                self.pasos_frames[paso // pasos_muestreo_frames] = paso

            if calcular_magnetizacion and paso % pasos_muestreo_magnetizacion == 0:
                magnetizacion = np.sum(self.web) / (self.NN)
                self.magnetizaciones[paso // pasos_muestreo_magnetizacion] = magnetizacion
                self.pasos_magnetizaciones[paso // pasos_muestreo_magnetizacion] = paso

            if generar_imagenes:
                if paso in pasos_imagenes:
                    self.imagenes[aux_idx] = self.web.copy()
                    self.pasos_imagenes[aux_idx] = paso
                    aux_idx += 1

            if calcular_energia and paso % pasos_muestreo_energia == 0:
                self.energia[paso // pasos_muestreo_energia] = self.calcular_energia()
                self.pasos_energia[paso // pasos_muestreo_energia] = paso
        
            
            # Genera los numeros aleatorios para todo el paso Monte Carlo
            # Mejora el rendimiento
            i_idx = rng.integers(0, self.N, size=self.NN)
            j_idx = rng.integers(0, self.N, size=self.NN)

            # Genera tambien los valores de aceptacion para cada espin
            aceptacion = rng.random(size=self.NN)


            for k in range(self.NN):
                i, j = i_idx[k], j_idx[k]
                p_aceptacion = aceptacion[k]

                prob = self.calcular_probabilidad(i, j)
                if p_aceptacion < prob:
                    # Cambia el espín
                    self.web[i, j] *= -1

        # Guardar (si necesario) el estado final

        if generar_gif and (len(self.frames) == 0 or self.pasos_frames[-1] != pasos_MC):
            self.frames[-1] = self.web.copy()
            self.pasos_frames[-1] = pasos_MC

        if calcular_magnetizacion and (len(self.magnetizaciones) == 0 or self.pasos_magnetizaciones[-1] != pasos_MC):
            magnetizacion = np.sum(self.web) / (self.NN)
            self.magnetizaciones[-1] = magnetizacion
            self.pasos_magnetizaciones[-1] = pasos_MC

        if generar_imagenes and (len(self.imagenes) == 0 or self.pasos_imagenes[-1] != pasos_MC):
            self.imagenes[-1] = self.web.copy()
            self.pasos_imagenes[-1] = pasos_MC

        if calcular_energia and (len(self.energia) == 0 or self.pasos_energia[-1] != pasos_MC):
            self.energia[-1] = self.calcular_energia()
            self.pasos_energia[-1] = pasos_MC

    def simular_precompute(self, pasos_MC=1000, generar_gif=False, pasos_muestreo_frames=100, generar_imagenes=True, n_imagenes=5, espaciado_imagenes='log', calcular_magnetizacion=True, pasos_muestreo_magnetizacion=100, calcular_energia=False, pasos_muestreo_energia=10):

        # Generador de números aleatorios
        rng = np.random.default_rng(seed=self.SEED)

        # Pre-aloca espacio para los resultados en numpy arrays (mas eficiente)
        # Si no se quiere guardar algun resultado, se inicializan sus arrays como None
        # esto es necesario por consistencia en el return

        # Arrays para los Gifs
        if generar_gif:
            n_frames = pasos_MC // pasos_muestreo_frames + 1
            self.frames = np.empty((n_frames, self.N, self.N), dtype=np.int8)
            self.pasos_frames = np.empty(n_frames, dtype=int)
        else:
            self.frames = None
            self.pasos_frames = None

        # Arrays para las magnetizaciones
        if calcular_magnetizacion:
            n_magnetizaciones = pasos_MC // pasos_muestreo_magnetizacion + 1
            self.magnetizaciones = np.empty(n_magnetizaciones, dtype=float)
            self.pasos_magnetizaciones = np.empty(n_magnetizaciones, dtype=int)
        else:
            self.magnetizaciones = None
            self.pasos_magnetizaciones = None

        # Energia
        if calcular_energia:
            n_energia = pasos_MC // pasos_muestreo_energia + 1
            self.energia = np.empty(n_energia, dtype=float)
            self.pasos_energia = np.empty(n_energia, dtype=int)
        else:
            self.energia = None
            self.pasos_energia = None

        # Arrays para las imagenes
        if generar_imagenes:
            self.imagenes = np.empty((n_imagenes, self.N, self.N), dtype=np.int8)
            self.pasos_imagenes = np.empty(n_imagenes, dtype=int)
            if espaciado_imagenes == 'lineal':
                pasos_imagenes = np.linspace(0, pasos_MC, n_imagenes, dtype=int)
            elif espaciado_imagenes == 'log':
                pasos_imagenes = np.logspace(0, np.log10(pasos_MC), n_imagenes, dtype=int)
            aux_idx = 0
        else:
            self.imagenes = None
            self.pasos_imagenes = None


        # Simulacion
        for paso in range(pasos_MC):
            # Guarda el estado inicial
            if generar_gif and paso % pasos_muestreo_frames == 0:
                self.frames[paso // pasos_muestreo_frames] = self.web.copy()
                self.pasos_frames[paso // pasos_muestreo_frames] = paso

            if calcular_magnetizacion and paso % pasos_muestreo_magnetizacion == 0:
                magnetizacion = np.sum(self.web) / (self.NN)
                self.magnetizaciones[paso // pasos_muestreo_magnetizacion] = magnetizacion
                self.pasos_magnetizaciones[paso // pasos_muestreo_magnetizacion] = paso

            if generar_imagenes:
                if paso in pasos_imagenes:
                    self.imagenes[aux_idx] = self.web.copy()
                    self.pasos_imagenes[aux_idx] = paso
                    aux_idx += 1

            if calcular_energia and paso % pasos_muestreo_energia == 0:
                self.energia[paso // pasos_muestreo_energia] = self.calcular_energia()
                self.pasos_energia[paso // pasos_muestreo_energia] = paso
        
            

            # Genera tambien los valores de aceptacion para cada espin
            aceptacion = rng.random(size=(self.N, self.N))

            # Obten los valores de dE para cada espin
            dE = self.compute_dE_all()

            # Genera spins aleatorios a evaluar
            i_idx = rng.integers(0, self.N, size=self.NN)
            j_idx = rng.integers(0, self.N, size=self.NN)

            selected_idx_mask = np.zeros((self.N, self.N), dtype=bool)
            selected_idx_mask[i_idx, j_idx] = True

            # Divide la red como un tablero de ajedrez para mejorar la convergencia
            # Alterna entre dos subredes (blancas y negras)
            mask_impares = (np.indices((self.N, self.N)).sum(axis=0) % 2 == 0)
            mask_pares = ~mask_impares

            for update_mask in [mask_impares, mask_pares]:

                # Los indices generados son los que se van a evaluar
                update_mask = np.logical_and(update_mask, selected_idx_mask)

                # Crear máscara de aceptación
                accept_mask = (dE <= 0) | (aceptacion < np.vectorize(self.exp_dE.get)(dE))

                final_mask = np.logical_and(update_mask, accept_mask)

                # Cambia los espines que cumplen la condición
                self.web[final_mask] *= -1
            

        # Guardar (si necesario) el estado final

        if generar_gif and (len(self.frames) == 0 or self.pasos_frames[-1] != pasos_MC):
            self.frames[-1] = self.web.copy()
            self.pasos_frames[-1] = pasos_MC

        if calcular_magnetizacion and (len(self.magnetizaciones) == 0 or self.pasos_magnetizaciones[-1] != pasos_MC):
            magnetizacion = np.sum(self.web) / (self.NN)
            self.magnetizaciones[-1] = magnetizacion
            self.pasos_magnetizaciones[-1] = pasos_MC

        if generar_imagenes and (len(self.imagenes) == 0 or self.pasos_imagenes[-1] != pasos_MC):
            self.imagenes[-1] = self.web.copy()
            self.pasos_imagenes[-1] = pasos_MC

        if calcular_energia and (len(self.energia) == 0 or self.pasos_energia[-1] != pasos_MC):
            self.energia[-1] = self.calcular_energia()
            self.pasos_energia[-1] = pasos_MC


    def precompute_all(self):
        """
        Precomputa los vecinos más cercanos y de siguiente nivel para cada espín en la red.
        Esto es más eficiente para simulaciones largas (muchos pasos Monte Carlo).
        """
        self.precomputar_vecinos_cercanos()
        self.precomputar_vecinos_diagonales()
        self.precomputar_exp_dE()

    def configurar_paths(self, data='data', gif_dir='gifs', imagenes_dir='images', magnetizaciones_dir='magnetizaciones'):
        """
        Configura los directorios para guardar los resultados de la simulación.
        - Parámetros:
            - gif_dir (str, opcional): El directorio para guardar los gifs. Por defecto es 'gifs'.
            - imagenes_dir (str, opcional): El directorio para guardar las imágenes. Por defecto es 'images'.
            - magnetizaciones_dir (str, opcional): El directorio para guardar las magnetizaciones. Por defecto es 'magnetizaciones'.
        """

        if self.test_mode == True:
            test_dir = Path(f'test_N{self.N}_T{self.T}')
            test_dir.mkdir(exist_ok=True, parents=True)

            data_dir = test_dir / data
            data_dir.mkdir(exist_ok=True, parents=True)
            gif_dir = test_dir / gif_dir
            gif_dir.mkdir(exist_ok=True, parents=True)
            imagenes_dir = test_dir / imagenes_dir
            imagenes_dir.mkdir(exist_ok=True, parents=True)
            magnetizaciones_dir = test_dir / magnetizaciones_dir
            magnetizaciones_dir.mkdir(exist_ok=True, parents=True)

        else:
            resultados_dir = Path(f'resultados_N{self.N}_T{self.T}')
            resultados_dir.mkdir(exist_ok=True, parents=True)

            data_dir = resultados_dir / data
            data_dir.mkdir(exist_ok=True, parents=True)
            gif_dir = resultados_dir / gif_dir
            gif_dir.mkdir(exist_ok=True, parents=True)
            imagenes_dir = resultados_dir / imagenes_dir
            imagenes_dir.mkdir(exist_ok=True, parents=True)
            magnetizaciones_dir = resultados_dir / magnetizaciones_dir
            magnetizaciones_dir.mkdir(exist_ok=True, parents=True)
            
        self.paths = {
            'data': data_dir,
            'gif': gif_dir,
            'imagenes': imagenes_dir,
            'magnetizaciones': magnetizaciones_dir
        }

    def warm_up(self, path_args=None, init_args=None):
        """
        Realiza un 'calentamiento' del modelo para establecer los paths correspondientes, inicializar la red. 
        """
        if path_args is not None:
            self.configurar_paths(**path_args)
        else:
            self.configurar_paths()


        if init_args is not None:
            self.inicia_red(**init_args)
        else:
            self.inicia_red()

    def calcular_calor_especifico(self, usar_ultimos_pasos=100):
        """
        Calcula el calor especifico mediante la desviación estandar de la energía.
        """
        if self.energia is None:
            raise ValueError("No se ha calculado la energía. Ejecuta la simulación primero.")


        E_std = np.std(self.energia[-usar_ultimos_pasos:])

        C = E_std**2 / (self.NN * self.T**2)
        self.C = C
        return C




    def crear_gif(self, nombre='ising.gif', intervalo=100, figsize=(10,6), dpi=150, dir_path=None):

        """
        Crea un archivo .gif animado desde la lista de fotogramas (matrices de spin).
        """
        if dir_path is None:
            dir_path = self.paths['gif']

        frames = self.frames
        custom_cmap = plt.get_cmap('coolwarm')

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        im = ax.imshow(frames[0], cmap=custom_cmap, vmin=-1, vmax=1)

        def update(i):
            im.set_data(frames[i])
            return im,

        file_path = dir_path / nombre
        ani = FuncAnimation(fig, update, frames=len(frames), interval=intervalo, blit=True)
        ani.save(file_path, dpi=dpi, writer=PillowWriter(fps=1000//intervalo))
        plt.close(fig)
      
    def crear_gif_rapido(self, nombre='ising_fast.gif', intervalo=100, dir_path=None):
        """
        Crea un archivo GIF animado usando imageio, mucho más rápido que matplotlib.
        """
        if dir_path is None:
            dir_path = self.paths['gif']

        frames = self.frames
        custom_cmap = plt.get_cmap('coolwarm')
        images = []

        for frame in frames:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.axis("off")
            im = ax.imshow(frame, cmap=custom_cmap, vmin=-1, vmax=1)
            fig.canvas.draw()

            # Extrae el array de la imagen del canvas
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            images.append(image)
            plt.close(fig)

        # Guarda el GIF
        file_path = dir_path / nombre
        imageio.mimsave(file_path, images, duration=intervalo/1000)

    def crear_gif_pil(self, nombre='ising_ultrafast.gif', intervalo=100, dir_path=None):
        """
        Genera un GIF animado en blanco y negro usando PIL. Mucho más simple pero mucho más rápido.
        Ideal para visualizar redes muy grandes.
        """
        if dir_path is None:
            dir_path = self.paths['gif']

        frames = self.frames
        images = []

        for frame in frames:
            # Normalizar los valores de -1,1 a 0,255 para imagen
            img_array = ((frame + 1) * 127.5).astype(np.uint8)
            img = Image.fromarray(img_array, mode='L')  # 'L' para escala de grises
            images.append(img)

        file_path = dir_path / nombre
        images[0].save(
            file_path, save_all=True, append_images=images[1:], duration=intervalo, loop=0
        )

    def guardar_imagenes(self, nombre='ising', figsize=(10,6), dpi=150, dir_path=None):
        """
        Guarda las imágenes de la red en el directorio especificado.
        """
        if dir_path is None:
            dir_path = self.paths['imagenes']

        for i, imagen in enumerate(self.imagenes):
            file_path = dir_path / f"{nombre}_N{self.N}_T{self.T}_{i+1}.png"
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(imagen, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_title(f"Paso {self.pasos_imagenes[i]}")
            fig.savefig(file_path, dpi=dpi)
            plt.close(fig)

    def graficar_magnetizacion(self, nombre='curva_magnetizacion.png', dir_path=None):
        """
        Grafica la magnetización a lo largo de los pasos de Monte Carlo.
        """
        if dir_path is None:
            dir_path = self.paths['magnetizaciones']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.pasos_magnetizaciones, self.magnetizaciones, marker='o', linestyle='--', color='b')
        ax.set_title(f'Magnetización vs Pasos MC (N={self.N}, T={self.T})')
        ax.set_xlabel('Pasos Monte Carlo')
        ax.set_ylabel('Magnetización')
        ax.grid()
        file_path = dir_path / nombre
        fig.savefig(file_path, dpi=150)
        plt.close(fig)

    def guardar_magnetizacion(self, nombre='magnetizacion', dir_path=None):
        """
        Guarda la magnetización en un archivo de texto.
        """
        if dir_path is None:
            dir_path = self.paths['data']

        file_path = dir_path / f"{nombre}_N{self.N}_T{self.T}.txt"
        data = np.column_stack((self.pasos_magnetizaciones, self.magnetizaciones))
        np.savetxt(file_path, data, header="Pasos Monte Carlo\tMagnetización", fmt="%d\t%.6f")

    def guardar_energia(self, nombre='energia', dir_path=None):
        """
        Guarda la energía en un archivo de texto.
        """
        if dir_path is None:
            dir_path = self.paths['data']

        file_path = dir_path / f"{nombre}_N{self.N}_T{self.T}.txt"
        data = np.column_stack((np.arange(len(self.energia)), self.energia))
        np.savetxt(file_path, data, header="Pasos Monte Carlo\tEnergia", fmt="%d\t%.6f")