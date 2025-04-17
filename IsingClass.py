import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class IsingModel:

    def __init__(self, N, T, J1, J2, test_mode=False, SEED=None):
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
        self.T = T
        self.J1 = J1
        self.J2 = J2

        # Variables de control de la simulación
        self.test_mode = test_mode
        self.SEED = SEED


    def inicia_red(self, valores=[1, -1], padding_width=1, mode='wrap'):
        """
        Inicializa la red (lattice) para la simulación del modelo de Ising.
        - Parámetros:
            - valores (list, opcional): Una lista de posibles valores de espín para la red. 
                          Por defecto es [1, -1].
        - Atributos:
            - web (numpy.ndarray): La red utilizada para la simulación.
        """
        rng = np.random.default_rng(self.SEED)
        self.web = rng.choice(valores, size=(self.N, self.N))


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
        E = 2 * s[i, j] * (self.J1 * vecinos_inmediatos + self.J2 * vecinos_diagonales)

        return min(1, np.exp(-E / self.T))


    def simular(self, pasos_MC=1000, generar_gif=False, pasos_muestreo_frames=100, generar_imagenes=True, n_imagenes=5, espaciado_imagenes='lineal', calcular_magnetizacion=True, pasos_muestreo_magnetizacion=100):
        
        # Inicializa la red
        self.inicia_red()

        # Generador de números aleatorios
        rng = np.random.default_rng(seed=self.SEED)

        # Pre-aloca espacio para los resultados en numpy arrays (mas eficiente)
        # Si no se quiere guardar algun resultado, se inicializan sus arrays como None
        # esto es necesario por consistencia en el return

        # Arrays para los Gifs
        if generar_gif:
            n_frames = pasos_MC // pasos_muestreo_frames + 1
            self.frames = np.empty((n_frames, self.N, self.N), dtype=int)
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

        # Arrays para las imagenes
        if generar_imagenes:
            self.imagenes = np.empty((n_imagenes, self.N, self.N), dtype=int)
            self.pasos_imagenes = np.empty(n_imagenes, dtype=int)
            if espaciado_imagenes == 'lineal':
                pasos_imagenes = np.linspace(0, pasos_MC, n_imagenes, dtype=int)
            elif espaciado_imagenes == 'logaritmico':
                pasos_imagenes = np.logspace(0, pasos_MC, n_imagenes, dtype=int)
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
                magnetizacion = np.abs(np.sum(self.web) / (self.N * self.N))
                self.magnetizaciones[paso // pasos_muestreo_magnetizacion] = magnetizacion
                self.pasos_magnetizaciones[paso // pasos_muestreo_magnetizacion] = paso

            if generar_imagenes:
                if paso in pasos_imagenes:
                    self.imagenes[aux_idx] = self.web.copy()
                    self.pasos_imagenes[aux_idx] = paso
                    aux_idx += 1
        
            
            # Genera los numeros aleatorios para TODO el paso Monte Carlo
            # Mejora significativamente el rendimiento
            i_idx = rng.integers(0, self.N, size=self.N * self.N)
            j_idx = rng.integers(0, self.N, size=self.N * self.N)

            # Genera tambien los valores de aceptacion para cada espin
            aceptacion = rng.random(self.N * self.N)


            for k in range(self.N * self.N):
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
            magnetizacion = np.abs(np.sum(self.web) / (self.N * self.N))
            self.magnetizaciones[-1] = magnetizacion
            self.pasos_magnetizaciones[-1] = pasos_MC

        if generar_imagenes and (len(self.imagenes) == 0 or self.pasos_imagenes[-1] != pasos_MC):
            self.imagenes[-1] = self.web.copy()
            self.pasos_imagenes[-1] = pasos_MC

    def configurar_paths(self, gif_dir='gifs', imagenes_dir='images', magnetizaciones_dir='magnetizaciones'):
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

            gif_dir = test_dir / gif_dir
            gif_dir.mkdir(exist_ok=True, parents=True)
            imagenes_dir = test_dir / imagenes_dir
            imagenes_dir.mkdir(exist_ok=True, parents=True)
            magnetizaciones_dir = test_dir / magnetizaciones_dir
            magnetizaciones_dir.mkdir(exist_ok=True, parents=True)

        else:
            resultados_dir = Path(f'resultados_N{self.N}_T{self.T}')
            resultados_dir.mkdir(exist_ok=True, parents=True)
            gif_dir = resultados_dir / gif_dir
            gif_dir.mkdir(exist_ok=True, parents=True)
            imagenes_dir = resultados_dir / imagenes_dir
            imagenes_dir.mkdir(exist_ok=True, parents=True)
            magnetizaciones_dir = resultados_dir / magnetizaciones_dir
            magnetizaciones_dir.mkdir(exist_ok=True, parents=True)
            

        self.gif_dir = gif_dir
        self.imagenes_dir = imagenes_dir
        self.magnetizaciones_dir = magnetizaciones_dir


    def crear_gif(self, nombre='ising.gif', intervalo=100, dpi=150, dir_path=None):

        """
        Crea un archivo .gif animado desde la lista de fotogramas (matrices de spin).
        """
        if dir_path is None:
            dir_path = self.gif_dir

        frames = self.frames
        custom_cmap = plt.get_cmap('coolwarm')

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

        
    def guardar_imagenes(self, nombre='ising', dpi=150, dir_path=None):
        """
        Guarda las imágenes de la red en el directorio especificado.
        """
        if dir_path is None:
            dir_path = self.imagenes_dir

        for i, imagen in enumerate(self.imagenes):
            file_path = dir_path / f"{nombre}_N{self.N}_T{self.T}_{i+1}.png"
            plt.imsave(file_path, imagen, cmap='coolwarm', vmin=-1, vmax=1, dpi=dpi)
            plt.close()

    def graficar_magnetizacion(self, nombre='curva_magnetizacion.png', dir_path=None):
        """
        Grafica la magnetización a lo largo de los pasos de Monte Carlo.
        """
        if dir_path is None:
            dir_path = self.magnetizaciones_dir

        plt.figure(figsize=(10, 6))
        plt.plot(self.pasos_magnetizaciones, self.magnetizaciones, marker='o', linestyle='--', color='b')
        plt.title(f'Magnetización vs Pasos MC (N={self.N}, T={self.T})')
        plt.xlabel('Pasos Monte Carlo')
        plt.ylabel('Magnetización')
        plt.grid()
        file_path = dir_path / nombre
        plt.savefig(file_path, dpi=150)
        plt.close()