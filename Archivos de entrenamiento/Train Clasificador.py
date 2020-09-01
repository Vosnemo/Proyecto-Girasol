import sys # Librerias para desplazarse en carpetas
import os # Librerias para desplzarse en carpetas

import tensorflow as tf # Import tensorflow
import keras as keras #Import de keras
tf.compat.v1.disable_eager_execution() #Desabilitar el optimizador




from tensorflow.python.keras.preprocessing.image import ImageDataGenerator #Libraria para el preprocesamiento
from tensorflow.python.keras import optimizers #Optimizador para entrenar el algoritmo
from tensorflow.python.keras.models import Sequential #Redes neuronales secuenciales (Una de atras de la otra)
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D #Capas para las convoluciones y maxpuling
from tensorflow.python.keras import backend as K

K.clear_session() #Eliminar la sesion de keras


#Carpetas de imagenes
data_entrenamiento = './Clasificador/Entrenar' #Carpeta de entrenamiento
data_validacion = './Clasificador/Validar' #Carpeta de validacion

"""
Parameters
"""
epocas=15 #Epocas de entrenamiento
longitud, altura = 200, 200 #Nuevos tama;ios de las imagenes
batch_size = 50 #Numero de imagenes en cada paso
pasos = 1000 #Pasos de cada epoca
validation_steps = 100 #200 pasos de validacion en cada epoca
filtrosConv1 = 32 #Filtros en la convolucion 
filtrosConv2 = 64 #Filtros en cada convolucion
tamano_filtro1 = (3, 3) #Tama;io del filtro
tamano_filtro2 = (2, 2) #Tama;io del filtro
tamano_pool = (2, 2) #tama;o del max puling

clases = 2 #Clases por clasificar
lr = 0.004 #LearingRange ajuste de error


##Preparamos nuestras imagenes

#Preprocesamiento 1 
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255, #Reescalado de imagen. Valores de (0 1 ) 
    shear_range=0.2, #Reescalado de imagen y movimiento
    zoom_range=0.2, # Zoom de imagen
    horizontal_flip=True) #Inversion de imagenes

#-----Fin preprocesamiento 1


test_datagen = ImageDataGenerator(rescale=1. / 255) #Reescalado de imagen de validacion

# Procesamiento para todas las imagenes
entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento, #Entra a la carpeta de entrenamieno
    target_size=(altura, longitud), #Procesa todas las imagenes con la nueva longitud
    batch_size=batch_size, #Procesa todas las imagenes con el nuevo batch size
    class_mode='categorical') #Para la clasificaion categorica

# ---Fin procesamiento para todas las imagenes

#Procesamiento de imagenes de validacion 
validacion_generador = test_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
#--- Fin procesamiento de imagenes de validacion

#******************Inicio de red CNN


cnn = Sequential() #Capaca secuencial

##Primera convolucion (Num Filtros, Tam Filtro, Rezalizar en las esquinas, imagenes salen longitud y altura y RGB, variable de activacion)
cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))

#Despues de la primera capa de convolucion se tiene una de max pooling (tama;o de ^esta definido arriba el tama;o^)
cnn.add(MaxPooling2D(pool_size=tamano_pool))

## Segunda capa de convolucion
cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding ="same"))

# Segunda acapa de maxpooling
cnn.add(MaxPooling2D(pool_size=tamano_pool))

# ****Empieza la clasificacion******

cnn.add(Flatten()) #Hace que la imagen sea de una dimension con toda la informacion
cnn.add(Dense(256, activation='relu')) #Despues de aplanarla se manda a una capa de 256 neuronas y la funcion de activacion
cnn.add(Dropout(0.5)) #Apagarle el 50% de las neuronas para evitar el autoreajuste (caminos alternos para adaptar la informacion)
cnn.add(Dense(clases, activation='softmax')) #Capa que tiene las neuronas de las clases y calcula las probabilidades de que pueda ser una u otra

#Hace que durante el entrenamiento el algoritmo vea que tan bien va
#Funcion de perdida "Categorical crose..."
#Optimizador "Adam" Con el error definido arriba
#Metrica "accuaracy" Porcentaje de que tan bien aprende la red
cnn.compile(loss='categorical_crossentropy',
            optimizer=optimizers.Adam(lr=lr),
            metrics=['accuracy'])


#-------Fin de clasificacion


#****** INICIO Entrenamiento del algoritmo 
#Entrenar la red neuronal con (Imagen de entramiento
# Numero de pasos
# Numero de epocas
# Imagenes de validacion 
# Pasos por validacion  )

cnn.fit_generator(
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps)

#Corre mil pasos, luego corre 300 pasos de validacion y luego la otra epoca

#-------------Fin entrenamiento del algoritmo



#******Guardar el archivo de entrenamiento 

target_dir = './Entrenamiento Clasificador/' #Lugar donde guardar
if not os.path.exists(target_dir): #Si no existe la carpeta llamada modelo
  os.mkdir(target_dir) #Crea la carpeta 
cnn.save('./Entrenamiento Clasificador/modelo.h5') #Direccion donde guardar el modelo con nombre "modelo.h5"
cnn.save_weights('./Entrenamiento Clasificador/pesos.h5') #Direccion para guardar los pesos

#---Fin guardar el archivo