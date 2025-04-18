# Imports
import numpy as np
from IsingClass import IsingModel

# Esta es la seed. Si quieres reproducibilidad, ponla siempre igual.
# Si no te importa, puedes poner 'None'
SEED = 42


# Estos son los parametros del modelo clasico
params_clasico = {
    'N' : 64,
    'T' : 1.0,
    'J1' : 1.0,
    'J2' : 0.0,
    'seed' : SEED,
    'test_mode' : True
}

# Los parametros del modificado son iguales, solo hay que cambiar 'J2
params_mod = params_clasico.copy()
params_mod['J2'] = 2.0

# Aqui defines el modelo
# modelo_clasico = IsingModel(**params_clasico)   #este es el clasico
modelo_mod = IsingModel(**params_mod)           #este es el modificado

# Warmup
modelo_mod.warm_up()

# Aqui configuras los parametros de la simulacion
params_simulacion = {
    'pasos_MC' : 250,
    'generar_imagenes' : True,
    'n_imagenes' : 5,
    'espaciado_imagenes' : 'logaritmico',
    # Si quieres el gif tienes que descomentar las dos siguientes lineas
    # Y otras mas abajo indicadas igual.
    'generar_gif' : True,
    'pasos_muestreo_frames' : 10,
    'calcular_magnetizacion' : True,
    'pasos_muestreo_magnetizacion' : 5,
}


# Ejecuta la simulacion
# modelo_clasico.simular(**params_simulacion)
modelo_mod.simular(**params_simulacion)



# Guardar cosas del modelo clasico
# modelo_clasico.guardar_imagenes()
# modelo_clasico.graficar_magnetizacion()
# Si quieres el gif, descomenta la siguiente linea
# modelo_clasico.crear_gif() 


# Guardar cosas del modelo modificado
modelo_mod.guardar_imagenes()
modelo_mod.graficar_magnetizacion()
# Igual que antes, si quieres el gif, descomenta la siguiente linea
modelo_mod.crear_gif()