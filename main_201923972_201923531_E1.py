# %%LIBRERÍAS
import os
import glob
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from skimage.color import rgb2lab, rgb2hsv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from utils_201923972_201923531_E1 import JointColorHistogram, CatColorHistogram, kmeans_classifier
import seaborn as sns
sns.set()
print("Librarias cargadas")

# %% Cargamos las imagenes
train = [f for f in glob.glob(os.path.join('data_mp3\\DB\\train', '*.jpg'))]
test = [f for f in glob.glob(os.path.join('data_mp3\\DB\\test', '*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data_mp3\\DB\\Val', '*.jpg'))]
print("Número de imágenes en train cargadas:", len(train))
print("Número de imágenes en test cargadas:", len(test))
print("Número de imágenes en valid cargadas:", len(valid))

# %%
"""PROCEDIMIENTO"""

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
        # Se guarda el descriptor en un arreglo
        features.append(hist)
    features = np.asarray(features)

    # Guardamos features y labels en un csv
    labels_features = zip(labels, features)
    with open(route, 'w') as my_file:  # open the file in the write mode
        writer = csv.writer(my_file)  # create the csv writer
        writer.writerow(["labels", "features"])  # Write header of csv
        for data in labels_features:
            fila = [data[0]] + list(data[1])
            writer.writerow(fila)  # write a row to the csv file
    print('File \"{}\" created'.format(route))
    return features

print("Cargó la función color_201923972_201923531")


##  Anotaciones
#Creamos el array para guardar los labels de train
labels_train = np.array([Path(ruta).stem[:-3] for ruta in train])
unique_labels = ['Aechmea', 'cartucho', 'girasol', 'orquidea', 'rosa']
for i in range(len(labels_train)):
    if labels_train[i] not in unique_labels:
        if labels_train[i] =="orquide":
            labels_train[i] = "orquidea"

#Creamos el array para guardar los labels de valid
labels_valid = np.array([Path(ruta).stem[:-3] for ruta in valid])

##Experimentación
#Definimos el type, color_space y space_bins con los que se realizará la experimentación
list_Type = ["concat", "joint"]
list_color_space = ["rgb", "hsv", "lab"]
spaceBins = 10
unique_labels = ['Aechmea', 'cartucho', 'girasol', 'orquidea', 'rosa']
#Creamos el diccionario para guardar los resultados (predicciones) de cada experimento
dic_labels_predicted={}

for TYPE in list_Type: #Recorremos los tipos de histogramas
    for colorSpace in list_color_space: #Recorremos los espacios de color
        print("\nExperimento--> Type:{}, color_space:{}, space_bins:{}".format(TYPE, colorSpace, spaceBins))
        #Definimos la ruta para guardar los descriptores de color de train

        # --------------ENTRENAMIENTO-----------------
        route_train = 'data_mp3\\features_labels_train_{}_{}_{}.csv'.format(TYPE, colorSpace, spaceBins)
        # Hallamos los descriptores para train
        features_train = color_201923972_201923531(images=train, labels=labels_train, route=route_train, Type=TYPE,
                                                   space_bins=spaceBins, color_space=colorSpace)
        # Creamos el objeto de clase kmeans_classifier
        kMeans1 = kmeans_classifier()
        # Entrenamos a nuestro clasificador
        kMeans1.fit(features_train, labels_train)

        #----------------VALIDACIÓN----------------
        # Definimos la ruta para guardar los descriptores de color de valid
        route_valid = 'data_mp3\\features_labels_valid_{}_{}_{}.csv'.format(TYPE, colorSpace, spaceBins)
        # Hallamos los descriptores para valid
        features_valid = color_201923972_201923531(images=valid, labels=labels_valid, route=route_valid, Type=TYPE,
                                                   space_bins=spaceBins, color_space=colorSpace)
        # Obtenemos las predicciones(labels) para nuestras imagenes de valid
        predicted_labels = kMeans1.predict(features_valid)
        #Guardamos las predicciones(labels)
        dic_labels_predicted["{}_{}_{}".format(TYPE, colorSpace, spaceBins)] = predicted_labels

#%% Resultados Cuantitativos
if not os.path.exists('./data_mp3/ResultadosInforme/MatrizConfusion'):
    os.makedirs('./data_mp3/ResultadosInforme/MatrizConfusion')

df_general = pd.DataFrame()
df_general["Type"] = ["Precision", "Recall", "F1"]

#Recorremos el diccionario con los resultados(predicciones) de cada experimento
for key, predicted_labels in dic_labels_predicted.items():
        TYPE, colorSpace, spaceBins = key.split("_")
        titulo = "Experimento--> Type:{}, color_space:{}, space_bins:{}".format(TYPE, colorSpace, spaceBins)
        print("\n\x1b[0;35;40m" + titulo + '\x1b[0m\n')

        #Hallamos la matriz de confusión
        cm = confusion_matrix(y_true=labels_valid, y_pred=predicted_labels, labels=unique_labels)
        # Graficamos la matriz de confusión
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot()
        plt.grid(False)
        plt.suptitle(titulo)
        plt.savefig('./data_mp3/ResultadosInforme/MatrizConfusion/{}_{}_{}.png'.format(TYPE, colorSpace, spaceBins))
        plt.show()

        #------------Hallamos las métricas-----------------
        recall_clases = np.diag(cm) / np.sum(cm, axis=1) # Calculamos la cobertura
        # calculate precision using `zero_division` parameter
        precision_clases = np.diag(cm) / np.sum(cm, axis=0) # Calculamos la presición
        # calcular la precision asegurandose que no sean nan
        f1_clases = 2 * (precision_clases * recall_clases) / (precision_clases + recall_clases) # Calculamos la F medida
        print("\n\x1b[0;35;35m" + "Métricas del experimento por clase" + '\x1b[0m\n')
        df = pd.DataFrame()
        df["Type"] = unique_labels
        df["precision"] = precision_clases
        df["recall"] = recall_clases
        df["f1"] = f1_clases
        df.fillna(0, inplace=True)
        print(df)
        if not os.path.exists('./data_mp3/ResultadosInforme/Metricas'):
            os.makedirs('./data_mp3/ResultadosInforme/Metricas')
        df.to_csv("data_mp3/ResultadosInforme/Metricas/metricas_{}_{}_{}.csv".format(TYPE, colorSpace, spaceBins))
        print("\n\x1b[0;35;35m" + "Métricas generales del experimento" + '\x1b[0m\n')
        precision = precision_score(y_true=labels_valid, y_pred=predicted_labels, average='macro')
        recall = recall_score(y_true=labels_valid, y_pred=predicted_labels, average='macro')
        f1 = 2*(precision*recall)/(precision+recall)
        print(f"\nPrecision:{precision}", f"\nRecall:{recall}", f"\nF1:{f1}")
        df_general["{}_{}_{}".format(TYPE, colorSpace, spaceBins)] = [precision, recall, f1]

# save the results of the precision, recall and f1 in a csv
df_general.to_csv("./data_mp3/ResultadosInforme/Metricas/metricas_general.csv")

#%% Resultados Cualitativos
images = np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels

# Mejor Experimento
TYPE = "joint"
colorSpace = "hsv"
spaceBins = 10
predicted_labels = dic_labels_predicted["{}_{}_{}".format(TYPE, colorSpace, spaceBins)]

#Se halla el index de las imagenes bien clasificadas
index_clasificadas_bien = true_labels == predicted_labels

def Resultados_Cualitativos(index_clasificadas, images, true_labels, predicted_labels, columns):
    #Definimos el numero de filas del subplot en base a la cantidad de imagenes
    if np.sum(index_clasificadas) % columns == 0:
        rows = np.sum(index_clasificadas) // columns
    else:
        rows = np.sum(index_clasificadas) // columns + 1

    #Creamos la figura
    fig, ax = plt.subplots(nrows= rows, ncols=columns, figsize=(10,10))#Creamos la figura
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

#Realizamos una figura con las imagenes bien clasificadas
fig_bien = Resultados_Cualitativos(index_clasificadas_bien, images, true_labels, predicted_labels,columns = 3)
fig_bien.suptitle("Fortalezas del método Type:{}, color_space:{}, space_bins:{}".format(TYPE, colorSpace, spaceBins))
fig_bien.tight_layout()
if not os.path.exists('./data_mp3/ResultadosInforme/ResultadosCualitativos'):
    os.makedirs('./data_mp3/ResultadosInforme/ResultadosCualitativos')
fig_bien.savefig("./data_mp3/ResultadosInforme/ResultadosCualitativos/ResultadosCualitativosBien_{}_{}_{}.png".format(TYPE, colorSpace, spaceBins))
fig_bien.show()

#Se halla el index de las imagenes mal clasificadas
index_clasificadas_mal = np.invert(index_clasificadas_bien)
#Realizamos una figura con las imagenes mal clasificadas
fig_mal = Resultados_Cualitativos(index_clasificadas_mal, images, true_labels, predicted_labels, columns = 2)
fig_mal.suptitle("Debilidades del método Type:{}, color_space:{}, space_bins:{}".format(TYPE, colorSpace, spaceBins))
fig_mal.savefig("./data_mp3/ResultadosInforme/ResultadosCualitativos/ResultadosCualitativosMal_{}_{}_{}.png".format(TYPE, colorSpace, spaceBins))
fig_mal.show()

#%%Modelo Final: guardado
import joblib

def load_features_labels(filepath):
    """Función que carga el csv con features y labels"""
    file = open(filepath) #abrimos el archivo
    csvreader = csv.reader(file)
    header = next(csvreader) #leemos la primera linea
    features = []
    labels = []
    for row in csvreader: #leemos cada linea en el csv
        if len(row) != 0: #si la linea no esta vacia
            feature = row[1:]
            feature = [float(data) for data in feature]
            features.append(feature)
            labels.append(row[0])
    file.close()
    return np.array(features), np.array(labels)

# Mejor Experimento
TYPE = "joint"
colorSpace = "hsv"
spaceBins = 10
route_train = 'data_mp3\\features_labels_train_{}_{}_{}.csv'.format(TYPE, colorSpace, spaceBins)

# Cargamos el descriptor y los labels correspondientes al mejor experimento
features_train, labels_train = load_features_labels(route_train)

#Entrenamos el modelo
final_kmeans_model = kmeans_classifier().fit(features_train, labels_train)
# Guardamos el modelo entrenado
filename = "final_kmeans_model_201923972_201923531.pkl"
joblib.dump(final_kmeans_model, filename)
# Cargamos el modelo previamente guardado
final_kmeans_model = joblib.load(filename)

