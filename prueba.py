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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print("Librarias cargadas")

# %% Cargamos las imagenes
train = [f for f in glob.glob(os.path.join('data_mp3\\DB\\train', '*.jpg'))]
test = [f for f in glob.glob(os.path.join('data_mp3\\DB\\test', '*.jpg'))]
valid = [f for f in glob.glob(os.path.join('data_mp3\\DB\\Val', '*.jpg'))]
print("Número de imágenes en train cargadas:", len(train))
print("Número de imágenes en test cargadas:", len(test))
print("Número de imágenes en valid cargadas:", len(valid))
dic_images = {"train": train, "valid": valid, "test": test}
print("Imágenes cargadas")
# %%----------------------- Anotaciones--------------------------------
unique_labels = ['Aechmea', 'cartucho', 'girasol', 'orquidea', 'rosa']
# Creamos el array para guardar los labels de train
labels_train = np.array([Path(ruta).stem[:-3] for ruta in train])
for i in range(len(labels_train)):
    if labels_train[i] not in unique_labels:
        if labels_train[i] == "orquide":
            labels_train[i] = "orquidea"
# Creamos el array para guardar los labels de valid
labels_valid = np.array([Path(ruta).stem[:-3] for ruta in valid])
# Creamos el array para guardar los labels de test
labels_test = np.array([Path(ruta).stem[:-3] for ruta in test])
dic_labels = {"train": labels_train, "valid": labels_valid, "test": labels_test}  # diccionario de anotaciones
print("Anotaciones cargadas")
# %%----------------------- Carpetas/Rutas--------------------------------
if not os.path.exists('./data_mp3/texton_dictionary'): os.makedirs('./data_mp3/texton_dictionary')
if not os.path.exists('./data_mp3/texture'): os.makedirs('./data_mp3/texture')
if not os.path.exists('./data_mp3/shape'): os.makedirs('./data_mp3/shape')
if not os.path.exists('./data_mp3/ResultadosInforme'): os.makedirs('./data_mp3/ResultadosInforme')
if not os.path.exists('./data_mp3/Modelos'): os.makedirs('./data_mp3/Modelos')
# %%Descriptores de color
def color_201923972_201923531(images, labels, route, Type, space_bins, color_space):
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
        features.append(hist)  # Se guarda el descriptor en un arreglo
    features = np.asarray(features)
    if route != None:  # Si la ruta es especificada, se guarda los features y labels en un archivo .mat
        dic_labels_features = {"labels": labels, "features": features}
        scipy.io.savemat(route, dic_labels_features)  # guardamos los labels y features (texture)
        print('-->Archivo \"{}\" creado'.format(route))
    return features


print("Cargó la función color_201923972_201923531")
# %%Descriptores de textura
def calculate_filter_response_201923972_201923531(img, filters):
    all_filter_responses = []  # arreglo donde se guardará la respuesta a los filtros
    for i in range(filters.shape[2]):  # recorremos cada filtro en el banco de filtros
        Filter = filters[:, :, i]
        img_filter_response = correlate2d(img, Filter,
                                          mode="same")  # realizamos la cross-correlación entre la imagen y el filtro
        img_filter_response = img_filter_response.flatten()  # linealizamos la respuesta
        all_filter_responses.append(img_filter_response)  # guardamos la respuesta al filtro
    all_filter_responses = np.array(all_filter_responses)
    # Se crea el arreglo con la respuesta a los filtros de cada pixel
    responseByPixel = [all_filter_responses[:, j] for j in range(all_filter_responses.shape[1])]
    return responseByPixel

def calculate_texton_histogram_201923972_201923531(filter_response, dic_textons):
    k_bins = len(dic_textons.keys())  # definimos el número de bins/clusters
    textons = np.array(list(dic_textons.values()))
    # Se halla el texton mas "cercano" a cada pixel (según la distancia euclidean)
    textones_asociados = pairwise_distances_argmin(filter_response, textons, metric="euclidean")
    hist, _ = np.histogram(textones_asociados, bins=list(range(k_bins + 1)),
                           density=True)  # calculamos el histograma de textones
    return hist, _

def texture_201923972_201923531(images, labels, route, textons):
    features = []  # Se crea el arreglo para guardar los descriptores
    filters = scipy.io.loadmat('data_mp3\\filterbank.mat')["filterbank"]  # se carga el banco de filtros

    folder = None
    if "train" in images[0]: folder = "train"
    if "Val" in images[0]: folder = "valid"
    if "test" in images[0]: folder = "test"

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

    if route != None:  # Si la ruta es especificada, se guarda los features y labels en un archivo .mat
        dic_labels_features = {"labels": labels, "features": features}
        scipy.io.savemat(route, dic_labels_features)  # guardamos los labels y features (texture)
        print("-->descriptores de textura con {} bins para {} creados".format(len(textons.keys()), folder))
    return features
print("Se cargaron las funciones de texture")
#%%
def shape_201923972_201923531(images, labels, route, param1, param2):
    features = []
    img_matrix = []
    for i in range(0, len(images)):
        # Calculamos el descriptor de forma
        img = plt.imread(images[i])
        # Convertir a escala de grises
        img_gray = rgb2gray(img)  # al parecer también funciona image[:,:,0]
        descriptor, hog_image = hog(img_gray, orientations=param1, pixels_per_cell=(param2, param2),
                                    cells_per_block=(1, 1), visualize=True)
        # Se guarda el descriptor en un arreglo
        features.append(descriptor)
        # Se guarda la imagen en un arreglo
        img_matrix.append(hog_image)
    features = np.asarray(features)
    if route != None:  # Si la ruta es especificada, se guarda los features y labels en un archivo .mat
        dic_labels_features = {"labels": labels, "features": features}
        scipy.io.savemat(route, dic_labels_features)  # guardamos los labels y features (shape) en un archivo .mat
        print('-->Archivo \"{}\" creado'.format(route))
    return features
print("Función shape_201923972_201923531 cargada")
# %% Resultados Cualitativos

def Resultados_Cualitativos(index_clasificadas, images, true_labels, predicted_labels, columns, figsize=(10, 10)):
    # Definimos el numero de filas del subplot en base a la cantidad de imagenes
    if np.sum(index_clasificadas) % columns == 0:
        rows = np.sum(index_clasificadas) // columns
    else:
        rows = np.sum(index_clasificadas) // columns + 1

    # Creamos la figura
    fig, ax = plt.subplots(nrows=rows, ncols=columns, figsize=figsize)  # Creamos la figura
    ax = ax.ravel()
    c = 0

    for i in range(len(images)):  # Recorremos las imagenes
        if index_clasificadas[i]:  # Si la imagen esta en la lista de index_clasificadas
            ax[c].set_title("\nAnotación:{}, \nPredicción:{}".format(true_labels[i], predicted_labels[i]))
            ax[c].imshow(plt.imread(images[i]))
            ax[c].axis("Off")
            c += 1
    # Rellenamos con "imagenes en blanco" los subplots vacios
    while c < (rows * columns):
        ax[c].set_title("")
        ax[c].imshow(np.ones((3, 3, 3)))
        ax[c].axis("Off")
        c += 1
    return fig
print("Función Resultados_Cualitativos cargada")
# %% Resultados Cuantitativos
def Resultados_Cuantitativos(true_labels, predicted_labels, unique_labels, name_experiment):
    # Hallamos la matriz de confusión
    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels, labels=unique_labels)
    # ------------Calculamos las métricas-----------------
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
    # --------------- Graficamos la matriz de confusión--------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot()
    plt.grid(False)
    plt.suptitle(name_experiment)
    plt.show()
    continuar = input("\033[1;36m" + "See Plot of Confusion Matrix and Press Enter to continue..." + '\x1b[0m')
    return df, precision, f1, recall
print("Función Resultados_Cuantitativos cargada")
# %% Descriptores (color, texture, shape) para experimentación variando numero de clusters

TYPE = "joint"; colorSpace = "hsv"; spaceBins = 10  # Mejor Experimento color
bins_textons = 12  # Mejor experimento textura
best_texton_dictionary = np.load(os.path.join("data_mp3", "texton_dictionary", "texton_dictionary_{}.npy".format(bins_textons)),allow_pickle=True).item()
param1 = 32; param2 = 32  # Mejor experimemnto shape
dic_features = {"train": {}, "valid": {}}  # diccionario de features
lst_folder = ["train", "valid"]
for folder in lst_folder:
    print("\n\x1b[1;35m" + "Calculando descriptores para {}...".format(folder) + '\x1b[0m\n')
    route_color = "features_labels_{}_{}_{}_{}".format(folder, TYPE, colorSpace, spaceBins)
    route_texture = "features_labels_{}_{}".format(bins_textons, folder)
    route_shape = "features_labels_{}_orientation{}_pixelCell{}".format(folder,param1, param2)
    # hallamos los descriptores (feaures) para el mejor experimento
    dic_features[folder] = {
        "color": color_201923972_201923531(images=dic_images[folder], labels=dic_labels[folder], route=route_color, Type=TYPE,
                                           space_bins=spaceBins, color_space=colorSpace),
        "texture": texture_201923972_201923531(images=dic_images[folder], labels=dic_labels[folder], route=None,
                                               textons=best_texton_dictionary),
        "shape": shape_201923972_201923531(images=dic_images[folder], labels=dic_labels[folder], route=None,
                                           param1=param1, param2=param2)}
#%% Experimentacion clasificadores con parametros por defecto
clasificadores = {"SVM": SVC(), "RF": RandomForestClassifier()}
df_general = pd.DataFrame(columns=["Clasificador", "Descriptor", "Precisión", "Cobertura", "F1"],index=range(6))
dic_labels_predicted = {}  # diccionario donde se guardará las predicciones de cada experimento
i = 0
for name, clasificador in clasificadores.items():
    for feature in dic_features["train"]:
        titulo = "Experimento: clasificador {} con descriptor {}".format(name, feature)
        print("\n\x1b[1;35m" + titulo + '\x1b[0m\n')
        clasificador.fit(X=dic_features["train"][feature], y=dic_labels["train"])# ENTRENAMIENTO
        predicted_labels = clasificador.predict(X=dic_features["valid"][feature]) #VALIDACION
        dic_labels_predicted[name + "_" + feature] = predicted_labels #GUARDAMOS LAS PREDICCIONES
        df, precision, f1, recall =Resultados_Cuantitativos(true_labels=dic_labels["valid"],
                                                                 predicted_labels=predicted_labels,
                                                                 unique_labels=unique_labels, name_experiment=titulo) #RESULTADOS
        df_general.loc[i] = [name, feature, precision, recall, f1] #RESULTADOS GENERALES
        i += 1
print("\n\x1b[1;35;47m" + "Métricas generales experimentación clasificadores automaticos" + '\x1b[0m\n')
print(df_general)
df_general.to_csv("./data_mp3/ResultadosInforme/metricas_general_auto.csv")
#%% Experimento variando parámetros C y gamma de SVM

C = [1e3, 5e3, 1e4, 5e4, 1e5]
gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]
df_general = pd.DataFrame(columns=["Experimento", "Precisión", "Cobertura", "F1"], index=range(len(C) * len(gamma)*3))
dic_labels_predicted = {}  # diccionario donde se guardará las predicciones de cada experimento
i = 0
for feature in dic_features["train"]:
    for c in C:
        for g in gamma:
            titulo = "SVM para descriptor de {} con C={} y gamma={}".format(feature, c, g)
            print("\n\x1b[1;35m" + "Experimento: " + titulo + '\x1b[0m\n')
            # --------------ENTRENAMIENTO-----------------
            features_train = dic_features["train"][feature]
            clasificador = SVC(C=c, gamma=g, kernel='rbf', class_weight='balanced',random_state=42)
            clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
            print("-->Classicador entrenado")
            # ----------------VALIDACIÓN----------------
            features_valid = dic_features["valid"][feature]
            predicted_labels = clasificador.predict(features_valid)  # Obtenemos las predicciones(labels) para nuestras imagenes de valid
            dic_labels_predicted["{}_{}_{}".format(feature,c,g)] = predicted_labels  # Guardamos las predicciones(labels)
            print("-->Validación terminada")

            df, precision, f1, recall = Resultados_Cuantitativos(true_labels=dic_labels["valid"],
                                                                 predicted_labels=predicted_labels,
                                                                 unique_labels=unique_labels, name_experiment=titulo)
            df_general.iloc[i] = ["{}_{}_{}".format(feature,c,g), precision, recall, f1]
            i += 1

#%% Resultados Cualitativos y Cuantitativos SVM
print("\n\x1b[1;35;47m" + "Métricas generales de experimentación con SVM variando parámetros C y gamma" + '\x1b[0m\n')
print(df_general)
df_general.to_csv("./data_mp3/ResultadosInforme/metricas_general_SVM.csv")

images= np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels
feature = "color"; cp = 5e4; gam= 0.1 #Mejor experiemnto
predicted_labels = dic_labels_predicted["{}_{}_{}".format(feature, cp,gam)]
index_clasificadas_bien = true_labels == predicted_labels #Se halla el index de las imagenes bien clasificadas
dic = {"Fortalezas": index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }

for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = 3)
    fig.suptitle("Validación: {} del método Type:{}, color_space:{} con c:{} y gamma:{}".format(key, TYPE, colorSpace, cp, gam))
    fig.tight_layout()
    #fig.savefig("./data_mp3/ResultadosInforme/{}_shape_orientation{}_pixelCell{}.png".format(key,param1,param2))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')
# Guardamos el modelo entrenado del mejor experimento
clasificador = SVC(C=cp, gamma=gam, kernel='rbf', class_weight='balanced',random_state=42)
clasificador.fit(features_train, dic_labels["train"])
joblib.dump(clasificador, './data_mp3/Modelos/final_SVM_model_201923972_201923531.pkl')

#%% Experimento variando parámetros n_estimators y max_features para RandomForest
n_estim = [10, 50, 100, 200, 500]
max_feat = ["auto", "sqrt", "log2", None]
df_gen = pd.DataFrame(columns=["Experimento", "Precisión", "Cobertura", "F1"], index=range(len(n_estim) * len(max_feat)*3))
dic_labels_pred = {}  # diccionario donde se guardará las predicciones de cada experimento
j = 0
for feature in dic_features["train"]:
    for n in n_estim:
        for feat in max_feat:
            print("\n\x1b[1;35m" + "Calculando RF para {} con n_estimator={} y max_feature={}...".format(feature,n,feat) + '\x1b[0m\n')
            titulo = "RandomForest: n_estimators={}, max_features={}".format(n, feat)
            print("\033[1;35m" + titulo + '\x1b[0m\n')
            # --------------ENTRENAMIENTO-----------------
            features_train = dic_features["train"][feature]
            clasificador = RandomForestClassifier(n_estimators=n, max_features=feat, random_state=42)
            clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
            print("-->Classicador entrenado")
            # ----------------VALIDACIÓN----------------
            features_valid = dic_features["valid"][feature]
            predicted_labels = clasificador.predict(features_valid)  # Obtenemos las predicciones(labels) para nuestras imagenes de valid
            dic_labels_pred["{}_{}_{}".format(feature, n, feat)] = predicted_labels  # Guardamos las predicciones(labels)
            print("-->Validación terminada")

            df, precision, f1, recall = Resultados_Cuantitativos(true_labels=dic_labels["valid"],
                                                                 predicted_labels=predicted_labels,
                                                                 unique_labels=unique_labels, name_experiment=titulo)
            df_gen.iloc[j] = ["{}_{}_{}".format(feature, n, feat), precision, recall, f1]
            j+= 1
# Guardamos el modelo entrenado del mejor experimento
clasificador = RandomForestClassifier(n_estimators=n, max_features=feat, random_state=42)
clasificador.fit(features_train, dic_labels["train"])
joblib.dump(clasificador, './data_mp3/Modelos/final_RF_model_201923972_201923531.pkl')
#%% Resultados Cualitativos y Cuantitativos
print("\n\x1b[1;35;47m" + "Métricas generales de EXPERIMENTACIÓN: NÚMERO DE CLUSTERS" + '\x1b[0m\n')
print(df_gen)
df_gen.to_csv("./data_mp3/ResultadosInforme/metricas_general_baseline.csv")

images= np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels
feature = "color"; estim = 50; mFeat= "auto" #Mejor experiemnto
predicted_labels = dic_labels_pred["{}_{}_{}".format(feature, estim,mFeat)]
#Se halla el index de las imagenes bien clasificadas
index_clasificadas_bien = true_labels == predicted_labels
dic ={"Fortalezas":index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }

for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = 3)
    fig.suptitle("Validación: {} del método Type:{}, color_space:{} con c:{} y gamma:{}".format(key, TYPE, colorSpace, estim, mFeat))
    fig.tight_layout()
    #fig.savefig("./data_mp3/ResultadosInforme/{}_shape_orientation{}_pixelCell{}.png".format(key,param1,param2))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')
#%% Test
