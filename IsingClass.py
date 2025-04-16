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
        Parámetros:
            valores (list, opcional): Una lista de posibles valores de espín para la red. 
                          Por defecto es [1, -1].
            padding_width (int, opcional): El ancho del padding que se añadirá alrededor de la red. 
                           Por defecto es 1.
            mode (str, opcional): El modo utilizado para el padding. Por defecto es 'wrap', que 
                      envuelve los bordes de la red para crear condiciones de frontera periódicas.
        Atributos:
            web (numpy.ndarray): La red con padding utilizada para la simulación.
        """
        rng = np.random.default_rng(self.SEED)
        red = rng.choice(valores, size=(self.N, self.N))
        self.web = np.pad(red, pad_width=padding_width, mode=mode)


    def actualizar_contornos(self):
        """
        Actualiza los contornos de la red para mantener las condiciones de frontera periódicas.
        """
        s = self.web[1:-1, 1:-1]
        # Actualiza los bordes de la red
        self.web = np.pad(s, pad_width=1, mode='wrap')

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

        # Suma de vecinos inmediatos
        vecinos_inmediatos = s[i+1, j] + s[i-1, j] + s[i, j+1] + s[i, j-1]
        # Suma de vecinos diagonales
        vecinos_diagonales = s[i+1, j+1] + s[i-1, j+1] + s[i+1, j-1] + s[i-1, j-1]

        # Calculo del cambio de energía
        E = 2 * s[i, j] * (self.J1 * vecinos_inmediatos + self.J2 * vecinos_diagonales)

        return min(1, np.exp(-E / self.T))
    
    def extraer_red(self):
        """
        Extrae la red sin el padding.
        Devuelve:
            numpy.ndarray: La red sin el padding.
        """
        return self.web[1:-1, 1:-1]


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
        else:
            self.imagenes = None
            self.pasos_imagenes = None


        # Simulacion
        for paso in range(pasos_MC):
            # Guarda el estado inicial
            if generar_gif and paso % pasos_muestreo_frames == 0:
                self.frames[paso // pasos_muestreo_frames] = self.extraer_red()
                self.pasos_frames[paso // pasos_muestreo_frames] = paso

            if calcular_magnetizacion and paso % pasos_muestreo_magnetizacion == 0:
                magnetizacion = np.abs(np.sum(self.extraer_red()) / (self.N * self.N))
                self.magnetizaciones[paso // pasos_muestreo_magnetizacion] = magnetizacion
                self.pasos_magnetizaciones[paso // pasos_muestreo_magnetizacion] = paso

            if generar_imagenes:
                if espaciado_imagenes == 'lineal':
                    if paso % (pasos_MC // n_imagenes) == 0:
                        self.imagenes[paso // (pasos_MC // n_imagenes)] = self.extraer_red()
                        self.pasos_imagenes[paso // (pasos_MC // n_imagenes)] = paso
                elif espaciado_imagenes == 'logaritmico':
                    if paso % (int(pasos_MC / (np.log10(paso + 1)))) == 0:
                        self.imagenes[paso // (int(pasos_MC / (np.log10(paso + 1))))] = self.extraer_red()
                        self.pasos_imagenes[paso // (int(pasos_MC / (np.log10(paso + 1))))] = paso
            
            # Genera los numeros aleatorios para TODO el paso Monte Carlo
            # Mejora significativamente el rendimiento
            i_idx = rng.integers(1, self.N + 1, size=self.N * self.N)
            j_idx = rng.integers(1, self.N + 1, size=self.N * self.N)

            # Genera tambien los valores de aceptacion para cada espin
            aceptacion = rng.random(self.N * self.N)


            for k in range(self.N * self.N):
                i, j = i_idx[k], j_idx[k]
                p_aceptacion = aceptacion[k]

                prob = self.calcular_probabilidad(i, j)
                if p_aceptacion < prob:
                    # Cambia el espín
                    self.web[i, j] *= -1

            # Actualiza las condiciones de contorno una vez cada paso Monte Carlo
            self.actualizar_contornos()

        # Guardar (si necesario) el estado final

        if generar_gif and (len(self.frames) == 0 or self.pasos_frames[-1] != pasos_MC):
            self.frames[-1] = self.extraer_red()
            self.pasos_frames[-1] = pasos_MC

        if calcular_magnetizacion and (len(self.magnetizaciones) == 0 or self.pasos_magnetizaciones[-1] != pasos_MC):
            magnetizacion = np.abs(np.sum(self.extraer_red()) / (self.N * self.N))
            self.magnetizaciones[-1] = magnetizacion
            self.pasos_magnetizaciones[-1] = pasos_MC

        if generar_imagenes and (len(self.imagenes) == 0 or self.pasos_imagenes[-1] != pasos_MC):
            self.imagenes[-1] = self.extraer_red()
            self.pasos_imagenes[-1] = pasos_MC

    def configurar_paths(self, gif_dir='gifs', imagenes_dir='images', magnetizaciones_dir='magnetizaciones'):
        """
        Configura los directorios para guardar los resultados de la simulación.
        Parámetros:
            gif_dir (str, opcional): El directorio para guardar los gifs. Por defecto es 'gifs'.
            imagenes_dir (str, opcional): El directorio para guardar las imágenes. Por defecto es 'images'.
            magnetizaciones_dir (str, opcional): El directorio para guardar las magnetizaciones. Por defecto es 'magnetizaciones'.
        """

        if self.test_mode == True:
            test_dir = Path('test')
            test_dir.mkdir(exist_ok=True)

            gif_dir = test_dir / gif_dir
            gif_dir.mkdir(exist_ok=True)
            imagenes_dir = test_dir / imagenes_dir
            imagenes_dir.mkdir(exist_ok=True)
            magnetizaciones_dir = test_dir / magnetizaciones_dir
            magnetizaciones_dir.mkdir(exist_ok=True)

        else:
            gif_dir = Path(gif_dir)
            gif_dir.mkdir(exist_ok=True)
            imagenes_dir = Path(imagenes_dir)
            imagenes_dir.mkdir(exist_ok=True)
            magnetizaciones_dir = Path(magnetizaciones_dir)
            magnetizaciones_dir.mkdir(exist_ok=True)

        self.gif_dir = gif_dir
        self.imagenes_dir = imagenes_dir
        self.magnetizaciones_dir = magnetizaciones_dir


    def guardar_gif(self):
        """
        Guarda el gif de la simulación en el directorio especificado.
        """

        # Configura los paths
        self.configurar_paths()

        # Genera y guarda el gif
        if self.frames is not None:
            from PIL import Image
            frames = [Image.fromarray((frame + 1) * 127) for frame in self.frames]
            frames[0].save(self.gif_dir / f'Ising_N{self.N}_T{self.T}_J1{self.J1}_J2{self.J2}.gif', save_all=True, append_images=frames[1:], optimize=False, duration=100, loop=0)
        else:
            print("No hay frames para guardar.")