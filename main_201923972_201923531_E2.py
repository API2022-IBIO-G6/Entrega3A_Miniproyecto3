#%% LIBRERÍAS
import os
import glob
import csv
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, rgb2hsv, rgb2gray
from skimage.feature import hog
from skimage import exposure
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import scipy.io
from scipy.signal import correlate2d
from utils_201923972_201923531 import JointColorHistogram, CatColorHistogram, kmeans_classifier
import seaborn as sns; sns.set()
from tqdm import tqdm
print("Librarias cargadas")

#%% Cargamos las imagenes
train = [f for f in glob.glob(os.path.join('data_mp3\\DB\\train', '*.jpg'))]
test = [f for f in glob.glob(os.path.join('data_mp3\\DB\\test', '*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data_mp3\\DB\\Val', '*.jpg'))]
print("Número de imágenes en train cargadas:", len(train))
print("Número de imágenes en test cargadas:", len(test))
print("Número de imágenes en valid cargadas:", len(valid))
dic_images = {"train":train, "valid":valid, "test":test}

#%%----------------------- Anotaciones--------------------------------
unique_labels = ['Aechmea', 'cartucho', 'girasol', 'orquidea', 'rosa']
#Creamos el array para guardar los labels de train
labels_train = np.array([Path(ruta).stem[:-3] for ruta in train])
for i in range(len(labels_train)):
    if labels_train[i] not in unique_labels:
        if labels_train[i] =="orquide":
            labels_train[i] = "orquidea"
#Creamos el array para guardar los labels de valid
labels_valid = np.array([Path(ruta).stem[:-3] for ruta in valid])
#Creamos el array para guardar los labels de test
labels_test = np.array([Path(ruta).stem[:-3] for ruta in test])
dic_labels={"train":labels_train, "valid": labels_valid, "test": labels_test} #diccionario de anotaciones

#%%----------------------- Carpetas/Rutas--------------------------------
if not os.path.exists('./data_mp3/texton_dictionary'):os.makedirs('./data_mp3/texton_dictionary')
if not os.path.exists('./data_mp3/texture'):os.makedirs('./data_mp3/texture')
if not os.path.exists('./data_mp3/shape'):os.makedirs('./data_mp3/shape')
if not os.path.exists('./data_mp3/ResultadosInforme'):os.makedirs('./data_mp3/ResultadosInforme')

#%% Descriptores de textura (filter response)
def calculate_filter_response_201923972_201923531(img, filters):
    all_filter_responses = []  # arreglo donde se guardará la respuesta a los filtros
    for i in range(filters.shape[2]):  # recorremos cada filtro en el banco de filtros
        Filter = filters[:, :, i]
        img_filter_response = correlate2d(img, Filter, mode="same")  # realizamos la cross-correlación entre la imagen y el filtro
        img_filter_response = img_filter_response.flatten()  # linealizamos la respuesta
        all_filter_responses.append(img_filter_response)  # guardamos la respuesta al filtro
    all_filter_responses = np.array(all_filter_responses)
    # Se crea el arreglo con la respuesta a los filtros de cada pixel
    responseByPixel =[all_filter_responses[:, j] for j in range(all_filter_responses.shape[1])]
    return responseByPixel
#%% Filter response (train y valid)

filters = scipy.io.loadmat(os.path.join('data_mp3\\filterbank.mat'))["filterbank"] # se carga el banco de filtros
lst_folder = ["train", "valid"]
for folder in lst_folder:
    filter_response_by_img = []  # se crea una lista para guardar la respuesta a los filtros
    print("\n Calculando respuesta a los filtros en {}....".format(folder), "\nimagenes procesadas:")
    for i in tqdm(range(len(dic_images[folder]))):  # Se recorre cada ruta de la carpeta train
        img = plt.imread(dic_images[folder][i])  # la imagen se lee por defecto en rgb
        img_gris = rgb2gray(img)  # la imagen se convierte a escala de grises
        filter_response = calculate_filter_response_201923972_201923531(img_gris, filters)
        filter_response_by_img.append(filter_response)
    # Guardamos la respuesta a los filtros de cada uno de los pixels de las imagenes del folder
    np.save('data_mp3\\filter_response_{}.npy'.format(folder), filter_response_by_img)

#%% Descriptores de textura

def calculate_texton_dictionary_201923972_2019235312(filter_response, k, route_textons):
    model = KMeans(n_clusters=k, random_state=0)  # creamos nuestro clasificador (Kmeans)
    model.fit(filter_response)  # entrenamos el modelo
    centers = model.cluster_centers_  # obtenemos los centroides/textones
    texton_dictionary = dict(zip(range(k), centers))  # definimos el diccionario de textones
    np.save(route_textons, texton_dictionary)#se guarda el diccionario de textones
    return centers  # se retorna los centroides/textones

def calculate_texton_histogram_201923972_201923531(filter_response, dic_textons):
    k_bins = len(dic_textons.keys())  # definimos el número de bins/clusters
    textons= np.array(list(dic_textons.values()))
    # Se halla el texton mas "cercano" a cada pixel (según la distancia euclidean)
    textones_asociados = pairwise_distances_argmin(filter_response, textons, metric="euclidean")
    hist, _ = np.histogram(textones_asociados, bins=list(range(k_bins+1)), density=True)  # calculamos el histograma de textones
    return hist, _

def texture_201923972_201923531(images, labels, route, textons):
    features = []  # Se crea el arreglo para guardar los descriptores
    filters = scipy.io.loadmat('data_mp3\\filterbank.mat')["filterbank"]  # se carga el banco de filtros
    
    folder = None
    if "train" in images[0]:folder = "train"
    if "Val" in images[0]:folder = "valid"
    if "test" in images[0]:folder = "test"
    
    if os.path.exists("data_mp3\\filter_response_{}.npy".format(folder)):
        filter_response_by_img = np.load("data_mp3\\filter_response_{}.npy".format(folder))
        print("Respuesta a los filtros cargada desde archivo")
    else:
        filter_response_by_img = []  # se crea una lista para guardar la respuesta a los filtros
        print("\ncalculating filter response in {}....".format(folder), "\nimagenes procesadas:")
        for i in tqdm(range(len(images))):  # Se recorre cada ruta de la carpeta train
            img = plt.imread(images[i])  # la imagen se lee por defecto en rgb
            img_gris = rgb2gray(img)  # la imagen se convierte a escala de grises
            filter_response = calculate_filter_response_201923972_201923531(img_gris, filters)
            filter_response_by_img.append(filter_response)
    
    # se halla el histograma de textones
    for i in range(len(images)):  # Se recorre cada ruta de la lista images
        filter_response = filter_response_by_img[i]
        hist, _ = calculate_texton_histogram_201923972_201923531(filter_response, textons)  
        features.append(hist)  # guardamos el histograma
        
    if route != None: # Si la ruta es especificada, se guarda los features y labels en un archivo .mat
        dic_labels_features = {"labels": labels, "features": features}
        scipy.io.savemat(route, dic_labels_features)  # guardamos los labels y features (texture)
        print("-->descriptores de textura con {} bins para {} creados".format(len(textons.keys()),folder))
    return features

print("\ncargó la función texture_201923972_201923531")

#%% Experimentacion de número de textones que hay en el diccionario 

# Se carga la respuesta a los filtros de train (archivo .npy)
images_filter_response = np.load(os.path.join("data_mp3","filter_response_train.npy"))
print("Respuesta a los filtros en train (#images, #pixeles, #filtros) =",images_filter_response.shape)
all_filter_response_train =[]
for filter_response in images_filter_response:
    all_filter_response_train += list(filter_response)

list_k=[8,12,15,18]  # definimos el número de clusters/textons de la experimentación
texton_dictionaries=[]
print("\033[1;35m"+"\nCreando diccionarios de textones..."+ '\x1b[0m\n')
for k in list_k:
    route_texton_dictionary = "data_mp3\\texton_dictionary\\texton_dictionary_{}.npy".format(k)  # Definimos la ruta para guardar el modelo de textons
    centers = calculate_texton_dictionary_201923972_2019235312(all_filter_response_train, k, route_texton_dictionary)
    texton_dictionaries.append(route_texton_dictionary)
    print("-->diccionario con {} textones creado".format(k))

#%% Entrenamiento y Validación (TEXTURE FEATURES)

dic_labels_predicted_texture = {} #diccionario donde se guardará las predicciones de cada experimento
print("\n\x1b[1;35;47m" + "EXPERIMENTACIÓN DESCRIPTORES DE TEXTURA" + '\x1b[0m\n')
for route in texton_dictionaries:
    texton_dictionary = np.load(route, allow_pickle = True).item()
    k = len(texton_dictionary.keys())
    print("\033[1;35m"+"\nExperimento: {} bins/textones...".format(k)+ '\x1b[0m\n')

    # --------------ENTRENAMIENTO-----------------
    # Definimos la ruta para guardar los descriptores de textura de train
    route_train = 'data_mp3\\texture\\textones_labels_{}_train.mat'.format(k)
    # Hallamos los descriptores para train
    features_texture_train = texture_201923972_201923531(images=train, labels=labels_train, route=route_train,textons=texton_dictionary)
    clasificador = kmeans_classifier()  # Creamos el objeto de clase kmeans_classifier
    clasificador.fit(features_texture_train, labels_train)  # Entrenamos a nuestro clasificador
    print("-->Classicador entrenado")   
    
    # ----------------VALIDACIÓN----------------
    # Definimos la ruta para guardar los descriptores de textura de train
    route_valid = 'data_mp3\\texture\\textones_labels_{}_valid.mat'.format(k)
    # Hallamos los descriptores para valid
    features_texture_valid = texture_201923972_201923531(images=valid, labels=labels_valid, route=route_valid,textons=texton_dictionary)
    predicted_labels = clasificador.predict(features_texture_valid) # Obtenemos las predicciones(labels) para nuestras imagenes de valid
    dic_labels_predicted_texture[k] = predicted_labels #Guardamos las predicciones(labels)
    print("-->Validación terminada")
    
#%% Resultados Cuantitativos
def Resultados_Cuantitativos(true_labels, predicted_labels, unique_labels, name_experiment):
    #Hallamos la matriz de confusión
    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=unique_labels)
    #------------Calculamos las métricas-----------------
    recall_clases = np.diag(cm) / np.sum(cm, axis=1)  # Calculamos la cobertura
    precision_clases = np.diag(cm) / np.sum(cm, axis=0)  # Calculamos la presición
    f1_clases = 2 * (precision_clases * recall_clases) / (precision_clases + recall_clases)  # Calculamos la F medida
    print("\n\x1b[0;35;35m" + "Métricas del experimento por clase" + '\x1b[0m\n')
    df = pd.DataFrame()
    df["Type"] = unique_labels
    df["precision"] = precision_clases
    df["recall"] = recall_clases
    df["f1"] = f1_clases
    df.fillna(0, inplace=True)
    print(df)
    print("\n\x1b[0;35;35m" + "Métricas generales del experimento" + '\x1b[0m\n')
    precision = precision_score(y_true=labels_valid, y_pred=predicted_labels, average='macro')
    recall = recall_score(y_true=labels_valid, y_pred=predicted_labels, average='macro')
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"\nPrecision:{precision:.3f}", f"\nRecall:{recall:.3f}", f"\nF1:{f1:.3f}\n")
    #--------------- Graficamos la matriz de confusión--------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot()
    plt.grid(False)
    plt.suptitle(name_experiment)
    plt.show()
    continuar = input("\033[1;36m"+"See Plot of Confusion Matrix and Press Enter to continue..."+ '\x1b[0m')
    return df, precision, f1,  recall

#%%Resultados Cuantitativos de Texture
print("\n\x1b[1;35;47m" + "Resultados Cuantitativos" + '\x1b[0m\n')
df_texture = pd.DataFrame(columns=["bins/textones", "Precisión", "Cobertura", "F1"], index = range(4))
i = 0
#Recorremos el diccionario con los resultados(predicciones) de cada experimento
for k, predicted_labels in dic_labels_predicted_texture.items():
        titulo = "\nExperimento--> {} bins/textones...".format(k)
        print("\n\x1b[1;35m" + titulo + '\x1b[0m\n')
        df, precision, f1, recall = Resultados_Cuantitativos(true_labels=labels_valid, predicted_labels=predicted_labels, unique_labels=unique_labels, name_experiment = titulo)
        df_texture.iloc[i] = ["{}".format(k), precision, recall, f1]
        i += 1
print("\n\x1b[1;35;47m" + "Métricas generales de experimentación descriptores de textura" + '\x1b[0m\n', df_texture)
df_texture.to_csv("./data_mp3/ResultadosInforme/metricas_general_texture.csv")
#%% Resultados Cualitativos

def Resultados_Cualitativos(index_clasificadas, images, true_labels, predicted_labels, columns, figsize=(10,10)):
    #Definimos el numero de filas del subplot en base a la cantidad de imagenes
    if np.sum(index_clasificadas) % columns == 0:
        rows = np.sum(index_clasificadas) // columns
    else:
        rows = np.sum(index_clasificadas) // columns + 1

    #Creamos la figura
    fig, ax = plt.subplots(nrows= rows, ncols=columns, figsize=figsize)#Creamos la figura
    ax = ax.ravel()
    c = 0

    for i in range(len(images)): #Recorremos las imagenes
        if index_clasificadas[i]: #Si la imagen esta en la lista de index_clasificadas
            ax[c].set_title("\nAnotación:{}, \nPredicción:{}".format(true_labels[i], predicted_labels[i]))
            ax[c].imshow(plt.imread(images[i]))
            ax[c].axis("Off")
            c+= 1
    #Rellenamos con "imagenes en blanco" los subplots vacios
    while c < (rows*columns):
        ax[c].set_title("")
        ax[c].imshow(np.ones((3,3,3)))
        ax[c].axis("Off")
        c+= 1
    return fig

#%% Resultados Cualitativos de textura
images= np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels
bins_textons = 12 #Mejor experimento Textura
predicted_labels = dic_labels_predicted_texture[bins_textons]
best_texton_dictionary=np.load(os.path.join("data_mp3","texton_dictionary","texton_dictionary_{}.npy".format(bins_textons)), allow_pickle = True).item() 
#np.save("texton_dictionary_201923972_201923531.npy", best_texton_dictionary)#se guarda el diccionario de textones

#Se halla el index de las imagenes bien clasificadas
index_clasificadas_bien = true_labels == predicted_labels
dic ={"Fortalezas":index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }
for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = 3)
    fig.suptitle("{} del método con {} bins/textones".format(key, bins_textons))
    fig.tight_layout()
    #fig.savefig("./data_mp3/ResultadosInforme/{}_texture_{}.png".format(key, bins_textons))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')
    
#%% Descriptores de Forma
def shape_201923972_201923531(images, labels, route, param1, param2):
    """:param images:Imágenes de la cuales se calculará el descriptor
    :param labels:: Etiquetas de las imágenes.
    :param route: Ruta donde serán guardados los descriptores junto con las etiquetas.
    :param param1: : Parámetro 1 que variarán para calcular HOG.
    :param param2:Parámetro 2 que variarán para calcular HOG.
    :return: features: Arreglo con los descriptores de las imágenes (el mismo que se almacenará en
    route)."""
    features = []
    img_matrix = []
    for i in range(0,len(images)):
        # Calculamos el descriptor de forma
        img = plt.imread(images[i])
        # Convertir a escala de grises
        img_gray = rgb2gray(img) # al parecer también funciona image[:,:,0]
        descriptor, hog_image = hog(img_gray, orientations=param1, pixels_per_cell=(param2, param2),
                         cells_per_block=(1, 1), visualize=True)
        # Se guarda el descriptor en un arreglo
        features.append(descriptor)
        # Se guarda la imagen en un arreglo
        img_matrix.append(hog_image)
    features = np.asarray(features)
    if route != None: # Si la ruta es especificada, se guarda los features y labels en un archivo .mat
        dic_labels_features = {"labels": labels, "features": features}
        scipy.io.savemat(route, dic_labels_features)  # guardamos los labels y features (texture)
        print('-->Archivo \"{}\" creado'.format(route))
    return features
print("Función shape_201923972_201923531 cargada") 

#%%Experimentación de descriptores de forma

print("\n\x1b[1;35;47m" + "EXPERIMENTACIÓN DESCRIPTORES DE FORMA" + '\x1b[0m\n')
#Definimos 2 parámetros de la función para calcular HOG y 2 valores para cada uno de dichos parámetros
listParam1=[32,64]
listParam2=[32,64]
#Creamos el diccionario para guardar los resultados (predicciones) de cada experimento
dic_labels_predicted_shape={}
for param1 in listParam1:
    for param2 in listParam2:
        print("\033[1;35m"+"\nExperimento--> Orientation:{}, PixelxCell:{}".format(param1, param2)+ '\x1b[0m\n') 
        # --------------ENTRENAMIENTO-----------------
        # Definimos la ruta para guardar los descriptores de forma de train
        route_train = 'data_mp3\\shape\\features_labels_train_orientation{}_pixelCell{}.mat'.format(param1, param2)
        # Hallamos los descriptores de forma para train
        features_train = shape_201923972_201923531(images=train, labels=labels_train, route=route_train, param1=param1, param2=param2)
        clasificador = kmeans_classifier() # Creamos el objeto de clase kmeans_classifier
        clasificador.fit(features_train, labels_train) # Entrenamos a nuestro clasificador
        print("-->Classicador entrenado")
        
        # ----------------VALIDACIÓN-------------------
        # Definimos la ruta para guardar los descriptores de color de validación
        route_valid = 'data_mp3\\shape\\features_labels_valid_orientation{}_pixelCell{}.mat'.format(param1, param2)
        # Hallamos los descriptores para validación
        features_valid = shape_201923972_201923531(images=valid, labels=labels_valid, route=route_valid, param1=param1, param2=param2)
        # Obtenemos las predicciones(labels) para nuestras imagenes de valid
        labels_predicted = clasificador.predict(features_valid)
        # Guardamos las predicciones(labels) en un diccionario
        dic_labels_predicted_shape["{}_{}".format(param1,param2)] = np.array(labels_predicted)
        print("-->Validación terminada")

#%%Resultados Cuantitativos de Shape
print("\n\x1b[1;35;47m" + "Resultados Cuantitativos" + '\x1b[0m\n')
df_shape = pd.DataFrame(columns=["Orientacion","PixelXCelda","Precisión", "Cobertura", "F1"], index = range(4))
i = 0
#Recorremos el diccionario con los resultados(predicciones) de cada experimento
for experiment, predicted_labels in dic_labels_predicted_shape.items():
        param1, param2 = experiment.split("_")
        titulo = "Experimento--> orientation: {}, pixelCell: {}...".format(param1, param2)
        print("\n\x1b[1;35m" + titulo + '\x1b[0m\n')
        df, precision, f1, recall = Resultados_Cuantitativos(true_labels=labels_valid, predicted_labels=predicted_labels, unique_labels=unique_labels, name_experiment=titulo)
        df_shape.iloc[i]=[param1,param2, precision, recall, f1]
        i += 1
print("\n\x1b[1;35;47m" + "Métricas generales de experimentación descriptores de forma" + '\x1b[0m\n', df_shape)
df_shape.to_csv("./data_mp3/ResultadosInforme/metricas_general_shape.csv")

#%% Resultados Cualitativos de shape
images= np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels
param1 = 32; param2 = 32 #Mejor experiemnto shape
predicted_labels = dic_labels_predicted_shape["{}_{}".format(param1,param2)]

#Se halla el index de las imagenes bien clasificadas
index_clasificadas_bien = true_labels == predicted_labels
dic ={"Fortalezas":index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }
for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = 2)
    fig.suptitle("{} del método orientation: {}, pixelCell: {}".format(key, param1, param2))
    fig.tight_layout()
    #fig.savefig("./data_mp3/ResultadosInforme/{}_shape_orientation{}_pixelCell{}.png".format(key,param1,param2))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')

#%%Descriptores de color
def color_201923972_201923531(images, labels, route, Type, space_bins, color_space):
    """
    :param images: Lista con las imágenes (o rutas de las imágenes) a la cuales se les calculará el descriptor.
    :param labels:  Lista con las etiquetas de las imágenes pasadas por parámetro.
    :param route: Ruta donde serán guardados los descriptores junto con las etiquetas.
    :param Type:  Parámetro que indica el tipo de histograma de color que se calculará, puede ser “joint” o “concat”.
    :param space_bins: Número entero que indica el numero de bins que existirá por cada canal de color.
    :param color_space: Parámetro que indica el espacio de color que se usará, puede ser “rgb”, “hsv”, o “lab”.
    :return: Arreglo con los descriptores de las imágenes (el mismo que se almacenará en route)."""
    features = []  # Se crea el arreglo para guardar los descriptores
    # Se lee cada ruta de la lista images
    for ruta in images:
        # Se guarda la imagen en el espacio de color deseado
        img = plt.imread(ruta)  # la imagen se lee por defecto en rgb
        if color_space == "lab":
            img = rgb2lab(img)
        elif color_space == "hsv":
            img = rgb2hsv(img)
        # Calculamos el histograma de color de img según indique el parametro Type
        if Type == "joint":
            hist = JointColorHistogram(img, space_bins, min_val=None, max_val=None).flatten()
        elif Type == "concat":
            hist = CatColorHistogram(img, space_bins, min_val=None, max_val=None)
        else:
            print("Error: type must be 'joint' or 'concat'")
        features.append(hist)# Se guarda el descriptor en un arreglo
    features = np.asarray(features)
    if route != None: # Si la ruta es especificada, se guarda los features y labels en un archivo .mat
        dic_labels_features = {"labels": labels, "features": features}
        scipy.io.savemat(route, dic_labels_features)  # guardamos los labels y features (texture)
        print('-->Archivo \"{}\" creado'.format(route))
    return features

print("Cargó la función color_201923972_201923531")

#%% Descriptores (color, texture, shape) para experimentación variando numero de clusters

TYPE = "joint"; colorSpace = "hsv"; spaceBins = 10 # Mejor Experimento color
bins_textons = 12 #Mejor experimento textura
best_texton_dictionary=np.load(os.path.join("data_mp3","texton_dictionary","texton_dictionary_{}.npy".format(bins_textons)), allow_pickle = True).item() 
param1 = 32; param2 = 32 #Mejor experiemnto shape
dic_features={"train":{}, "valid": {}} #diccionario de features
lst_folder = ["train", "valid"]
for folder in lst_folder:
    print("\n\x1b[1;35m" + "Calculando descriptores para {}...".format(folder)+ '\x1b[0m\n')
    # hallamos los descriptores (feaures) para el mejor experimento
    dic_features[folder] = {"color":color_201923972_201923531(images=dic_images[folder], labels=dic_labels[folder], route=None, Type=TYPE,space_bins=spaceBins, color_space=colorSpace),
                    "texture":texture_201923972_201923531(images=dic_images[folder], labels=dic_labels[folder], route=None,textons=best_texton_dictionary),  
                    "shape": shape_201923972_201923531(images=dic_images[folder], labels=dic_labels[folder], route=None, param1=param1, param2=param2)}
    
#%%EXPERIMENTACIÓN: NÚMERO DE CLUSTERS
print("\n\x1b[1;35;47m" + "EXPERIMENTACIÓN: NÚMERO DE CLUSTERS" + '\x1b[0m\n')
list_k = [10,15,20,25] #Número de clusters a experimentar
df_general = pd.DataFrame(columns=["Experimento", "Precisión", "Cobertura", "F1"], index = range(4*3))
dic_labels_predicted = {} #diccionario donde se guardará las predicciones de cada experimento
i = 0
for feature in dic_features["train"]:
    for k in list_k:
        titulo = "\nExperimento: feature: {}, n_clusters:{}...".format(feature, k)
        print("\033[1;35m"+ titulo + '\x1b[0m\n')
        # --------------ENTRENAMIENTO-----------------
        features_train = dic_features["train"][feature]
        clasificador = kmeans_classifier(k)  # Creamos el objeto de clase kmeans_classifier
        clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
        print("-->Classicador entrenado")   
        # ----------------VALIDACIÓN----------------
        features_valid = dic_features["valid"][feature]
        predicted_labels = clasificador.predict(features_valid) # Obtenemos las predicciones(labels) para nuestras imagenes de valid
        dic_labels_predicted["{}_{}".format(feature, k)] = predicted_labels #Guardamos las predicciones(labels)
        print("-->Validación terminada")
        
        df, precision, f1,  recall= Resultados_Cuantitativos(true_labels=dic_labels["valid"], predicted_labels=predicted_labels, unique_labels=unique_labels, name_experiment=titulo)
        df_general.iloc[i]=["{}_{}".format(feature, k), precision, recall, f1]
        i += 1
        
#%% Resultados Cualitativos y Cuantitativos
print("\n\x1b[1;35;47m" + "Métricas generales de EXPERIMENTACIÓN: NÚMERO DE CLUSTERS" + '\x1b[0m\n')
print(df_general)
df_general.to_csv("./data_mp3/ResultadosInforme/metricas_general_baseline.csv")

images= np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels
feature = "color"; n_clusters = 20 #Mejor experiemnto
predicted_labels = dic_labels_predicted["{}_{}".format(feature, n_clusters)]
#Se halla el index de las imagenes bien clasificadas
index_clasificadas_bien = true_labels == predicted_labels
dic ={"Fortalezas":index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }

for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = 3)
    fig.suptitle("Validación: {} del método Type:{}, color_space:{} con {} clusters".format(key, TYPE, colorSpace, n_clusters))
    fig.tight_layout()
    #fig.savefig("./data_mp3/ResultadosInforme/{}_shape_orientation{}_pixelCell{}.png".format(key,param1,param2))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')

#%% Guardamos el modelo entrenado del mejor experimento
final_kmeans_model = kmeans_classifier(n_clusters).fit(dic_features["train"]["color"], labels_train)#Entrenamos el modelo
route_best_model = "final_kmeans_model_201923972_201923531.pkl"
#joblib.dump(final_kmeans_model, route_best_model)# Guardamos el modelo entrenado

#%%Test
true_labels=dic_labels["test"]
n_clusters = 20# clusters del mejor Experimento
print("\033[1;35m"+ "TEST" + '\x1b[0m\n')
if os.path.exists(route_best_model):
    clasificador = joblib.load(route_best_model)  # Cargamos el modelo previamente guardado
    print("Se cargó el mejor clasificador del archivo {}".format(route_best_model))
else:# ------------------------------ENTRENAMIENTO---------------------
    features_train = dic_features["train"]["color"]
    clasificador = kmeans_classifier(n_clusters)  # Creamos el objeto de clase kmeans_classifier
    clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
    print("-->Classicador entrenado")
# --------------------------------TEST--------------------------
features_test = color_201923972_201923531(images=dic_images["test"], labels=dic_labels["test"], route=None, Type=TYPE,space_bins=spaceBins, color_space=colorSpace)
predicted_labels = clasificador.predict(features_test) # Obtenemos las predicciones(labels) para nuestras imagenes de valid
print("-->Test terminado")
#------------Resultados Cualitativos y Cuantitativos---------------
df, precision, f1,  recall= Resultados_Cuantitativos(true_labels=dic_labels["test"], predicted_labels=predicted_labels, unique_labels=unique_labels, name_experiment="TEST")
#Se halla el index de las imagenes bien clasificadas
index_clasificadas_bien = true_labels == predicted_labels
dic ={"Fortalezas":index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }
for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, dic_images["test"], dic_labels["test"], predicted_labels,columns = 3)
    fig.suptitle("Test: {} del método Type:{}, color_space:{} con {} clusters".format(key, TYPE, colorSpace, n_clusters))
    fig.tight_layout()
    #fig.savefig("./data_mp3/ResultadosInforme/{}_test.png".format(key))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')