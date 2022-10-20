# %% 5. Creación de un modelo de red neuronal adecuado para el problema a analizar
import numpy as np
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL
from keras.utils.np_utils import to_categorical




import imutils

os.chdir("C:/deep_learning/proyecto_retino_2")

def RecorteContorno(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)


    
    # Umbralizamos la imagen (erosiones/dilataciones) para eliminar regiones
    # de ruido 
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh , None , iterations = 2)
    thresh = cv2.dilate(thresh , None , iterations = 2)
    

    # Busqueda de los contornos 
    cnts = cv2.findContours(thresh.copy(),
                            cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    
    # Usamos imutils 
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)

    # Obtener los puntos/pixeles extremos 
    extLeft = tuple(c[c[:,:,0].argmin()][0]) 
    extRight = tuple(c[c[:,:,0].argmax()][0]) 
    extTop = tuple(c[c[:,:,1].argmin()][0]) 
    extBot = tuple(c[c[:,:,1].argmax()][0]) 
    
    NuevoCerebro = image[ extTop[1]: extBot[1], extLeft[0]: extRight[0]]
    return NuevoCerebro


# %% Cargar los datos aumentados (crear objetos de tipo ndarray)
import numpy as np 
def jpg2Array(dir_list, image_size):
    
    """
    dir_list : lista de directorios : yes / no
    image_size = dimension de la imagen que va alimentar a mi red neuronal 
    
    """
    
    # Almacenar las variables independientes : X
    # Almacenar la variables dependiente : y
    
    # Creamos listas en blanco para almacenar las imagenes (ndarray)
    dataX = []
    datay = []
    
    # A partir de image_size debo obtener ancho y alto de las images que 
    # van a alimentar a mi red neuronal 
    image_width , image_height = image_size 
    
    # Usamos una estructura repetitiva para barrer los directorio yes/no
    for directory in dir_list:
        for filename in os.listdir(directory):
            # Cargamos la imagen 
            image = cv2.imread(directory + "/" + filename)
            
            # En este punto se necesita que la imagen almacena solo la informacion
            # de interes para alimentar el fit de mi red neuronal 
            image = RecorteContorno(image)
            
            # Redimensionamos deacuerdo a lo especificado en el argumento image_size
            image = cv2.resize(image , dsize = (image_width, image_height),
                               interpolation = cv2.INTER_CUBIC)

                         
            # Normalizamos los pixeles 
            #image  = image / 255
            
            # Almacenamos la imagen procesada
            imgNP_Red = np.array(image)[:,:,0] 
            imgNP_Green = np.array(image)[:,:,1]
            imgNP_Blue = np.array(image)[:,:,2]
            imgNP_Color = np.concatenate((imgNP_Red, imgNP_Green), axis=0)
            imgNP_Color = np.concatenate((imgNP_Color, imgNP_Blue), axis=0)

            dataX.append(imgNP_Color)

            if directory[-1:] == '0':
                datay.append([0])
            if directory[-1:] == '4':
                datay.append([1])
    
    # Transformamos las listas X e y a objetos de tipo ndrray
    X = np.array(dataX)

    X.shape
   
    y = np.array(datay)
    y.shape

   

    return X,y
    

# Probemos jpg2Array
# 
# Directorio donde estan las imagenes aumentadas 
augmented_data = os.getcwd()  + "/retino_dataset_f1"+ "/retino" + "/augmented_data"

# Creamos los nombres de directorios (imagenes aumentadas) para yes/no
augmented_0 = augmented_data + "/0"
augmented_4 = augmented_data + "/4"

# Definimos la dimension de las imagenes resultado 
IMG_WIDTH, IMG_HEIGHT = (520,520)


varX, vary = jpg2Array([augmented_0,augmented_4] , 
                       (IMG_WIDTH, IMG_HEIGHT))


varX

vary
# %% Particionamiento del conjunto de datos 
from sklearn.model_selection import train_test_split


train_X, test_X, train_y, test_y = model_selection.train_test_split(varX,vary,test_size=0.2)

# Visualicemos las imagenes
fig, ax =  plt.subplots(1,3, figsize = (10,10))
for i, axi in enumerate(ax.flat):
    axi.imshow(train_X[i])
    axi.set(xticks=[], yticks=[])
plt.show()

train_X  = train_X.reshape((train_X.shape[0], 520,520,3))
test_X  = test_X.reshape((test_X.shape[0], 520,520,3))

# Convertimos a punto flotante: a las variables independientes
train_X = train_X.astype("float32")
test_X = test_X.astype("float32")

train_X.shape
#Normalizamos
train_X = train_X/255
test_X = test_X/255

train_y = to_categorical(train_y, 2)
test_y = to_categorical(test_y, 2)



# %% Definicion del modelo
# Tiene dos aspecots fundamentales:
    # FRONT-END: Extracción de características (capas conv. y capas pooling)
    # BACK-END: Clasificador

# Capa convulucional: filtro de (3,3) y 32 filtros y un capa pooling(max_pooling)
# Necesitamos aplanar los objetos bidimensionales (resultado de la conv. y del pooling)

# En vista que tenemos un problema de clasificación (2 clases)
# ==> necesitamos una capa de salida con 2 nodos a predecir y una función de activacion sigmoid

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import gradient_descent_v2
from keras.models import load_model


model = Sequential()
# Primera capa oculta: Especificar los inputs
model.add(
    Conv2D(
        64, (2,2),
        activation="relu",
        input_shape = (520,520,3)
    )
)
model.add(MaxPooling2D((2,2)))
# Segunda capa oculta
model.add(
    Conv2D(
        128, (2,2),
        activation="relu"
    )
)
model.add(MaxPooling2D((2,2)))

# Podemos agregar mas capas: Conv2D + MaxPooling2D
model.add(Flatten())
model.add(Dense(1024, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Veamos algo de información que contiene la red neuronal

model.summary()


# %% 3er Paso: Compilar el modelo

# Funcion de perdida : cross entropy (binary_crossentropy)
# Definir el optimizador : adam 
# metrica : accuracy

# Compilacion del modelo 
model.compile(
    loss = "categorical_crossentropy",
    optimizer = 'adam',
    metrics = 'accuracy'
)


# %% 4to Paso : Ajustemos el modelo 
# Este ajuste ocurre en una serie de epocas (epochs) y cada epoca se va dividir
# en lotes (batch_size)
# 
# epoch : una pasada a traves de todas las filas del conjunto de datos de train
# batch_size : UNa o mas muestras (filas) consideradas por el modelo dentro de una epoca
# antes de que se actualicen los pesos 

inicio_CNN = time.time()
model.fit(train_X, train_y, batch_size= 100, epochs= 25)
final_CNN = time.time()
print("Tiempo de Ajuste del modelo CNN:\t",final_CNN-inicio_CNN) 

# %% 5to paso: Almacenamos el modelo y su arquitectura en un archivo 
model.save("13_CNN_320_320_3.h5")

_, precision = model.evaluate(train_X,train_y)

print("Precion del modelo train: %.2f" %(precision*100)) # 100.00

_, precision = model.evaluate(test_X,test_y)

print("Precion del modelo test: %.2f" %(precision*100)) # 100.00


