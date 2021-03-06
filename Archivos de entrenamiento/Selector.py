import numpy as np
import keras as keras #Import de keras
import h5py


from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model


from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.models import load_model
from tensorflow.keras.models import load_model



longitud, altura = 200, 200 #Longitud y altura anteriores
modelo = './Entrenamiento Selector/modelo.h5' #Directorio de modelo
pesos_modelo = './Entrenamiento Selector/pesos.h5' #Directorio De pesos
cnn = load_model(modelo) #Cargar modelo a la red neuronal
cnn.load_weights(pesos_modelo) #Cargar pesos a la red neuronal

def predict(file): #Procedimiento para predecir (Recibe la imagen)
  x = load_img(file, target_size=(longitud, altura)) #A la variable se le carga la imagen a predecir
  x = img_to_array(x) #Se convierte en un arreglo la imagen para obtener los valores
  x = np.expand_dims(x, axis=0) #En el 0 se a;ade una dimension extra para procesar la informacion
  array = cnn.predict(x) #Se llama a la red con la imagen cargada
  result = array[0] #Devuelve un arreglo de dos dimensiones, que tiene un solo valor con la dimension con toda la informacion 
  answer = np.argmax(result) #Respuesta es lo que tenga el resultado en esa capa
  #print (array)
  if answer == 0:
     print("prediccion: Semillas")
     if (result[0]<0.8):
           print("prediccion: Semillas")
           print("Fase 1")
           #print(result[0])
     print(result)
  elif answer == 1:
     print("prediccion: Girasol")
     print("Fase final")
     #print(result)
  elif answer == 2:
     print("prediccion: Boton")
     print("Fase 2: Tiempo 3 meses")
    # print(result)

  return answer

predict('./Pruebas/G8.jpg')

