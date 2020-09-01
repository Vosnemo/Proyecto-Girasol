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
modelo ='./Entrenamiento Clasificador/modelo.h5' #Directorio de modelo
pesos_modelo ='./Entrenamiento Clasificador/pesos.h5' #Directorio De pesos
cnn = load_model(modelo) #Cargar modelo a la red neuronal
cnn.load_weights(pesos_modelo) #Cargar pesos a la red neuronal

def predict(file): #Procedimiento para predecir (Recibe la imagen)
  x = load_img(file, target_size=(longitud, altura)) #A la variable se le carga la imagen a predecir
  x = img_to_array(x) #Se convierte en un arreglo la imagen para obtener los valores
  x = np.expand_dims(x, axis=0) #En el 0 se a;ade una dimension extra para procesar la informacion
  array = cnn.predict(x) #Se llama a la red con la imagen cargada
  result = array[0] #Devuelve un arreglo de dos dimensiones, que tiene un solo valor con la dimension con toda la informacion 
  answer = np.argmax(result) #Respuesta es lo que tenga el resultado en esa capa
  if answer == 0:
    print("Prediccion: No es Girasol")
    print(result[0])
    
  elif answer == 1:
      if (result[1]>0.8):
          print("Prediccion: Es Girasol")

      else:
          print(answer)  
          print("Prediccion:NO Es Girasol")
            


  return answer

predict('./Pruebas/E8.jpg')

