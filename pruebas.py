import numpy as np
from IsingClass import IsingModel

SEED = 42

params_clasico = {
    'N' : 64,
    'T' : 1.0,
    'J1' : 1.0,
    'J2' : 0.0,
    'SEED' : SEED,
    'test_mode' : True
}

params_mod = params_clasico.copy()
params_mod['J2'] = 2.0

# Aqui defines el modelo
modelo_clasico = IsingModel(**params_clasico)
# modelo_mod = IsingModel(**params_mod)

# Inicia la red
modelo_clasico.inicia_red()

# Aqui configuras los parametros de la simulacion
params_simulacion = {
    'pasos_MC' : 250,
    'generar_imagenes' : True,
    'n_imagenes' : 5,
    'espaciado_imagenes' : 'lineal',
    'generar_gif' : True,
    'pasos_muestreo_frames' : 10,
    'calcular_magnetizacion' : True,
    'pasos_muestreo_magnetizacion' : 5,
}

# Ejecuta la simulacion
modelo_clasico.simular(**params_simulacion)

# Configura los paths
modelo_clasico.configurar_paths()

# Guardar cosas
modelo_clasico.guardar_imagenes()
modelo_clasico.crear_gif()
modelo_clasico.graficar_magnetizacion()