#%% LIBRERÍAS
import os
import glob
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import scipy.io
from utils_201923972_201923531 import JointColorHistogram, CatColorHistogram, kmeans_classifier
import seaborn as sns; sns.set()
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

#%%----------------------- Carpetas/Rutas--------------------------------
if not os.path.exists('./data_mp3/texton_dictionary'): os.makedirs('./data_mp3/texton_dictionary')
if not os.path.exists('./data_mp3/texture'): os.makedirs('./data_mp3/texture')
if not os.path.exists('./data_mp3/shape'): os.makedirs('./data_mp3/shape')
if not os.path.exists('./data_mp3/ResultadosInforme'): os.makedirs('./data_mp3/ResultadosInforme')
if not os.path.exists('./data_mp3/Modelos'): os.makedirs('./data_mp3/Modelos')

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
    print("\n\x1b[0;35;35m" + "Métricas del experimento por clase" + '\x1b[0m')
    df = pd.DataFrame()
    df["Type"] = unique_labels
    df["precision"] = precision_clases
    df["recall"] = recall_clases
    df["f1"] = f1_clases
    df.fillna(0, inplace=True)
    print(df)
    print("\n\x1b[0;35;35m" + "Métricas generales del experimento" + '\x1b[0m')
    precision = precision_score(y_true=labels_valid, y_pred=predicted_labels, average='macro')
    recall = recall_score(y_true=labels_valid, y_pred=predicted_labels, average='macro')
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"\nPrecision: {precision:.3f}", f"\nRecall: {recall:.3f}", f"\nF1: {f1:.3f}")
    # --------------- Graficamos la matriz de confusión--------------------------
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot()
    plt.grid(False)
    plt.suptitle(name_experiment)
    plt.show()
    continuar = input("\n\x1b[4;93m" + "See Plot of Confusion Matrix and Press Enter to continue..." + '\x1b[0m')
    return df, precision, f1, recall
print("Función Resultados_Cuantitativos cargada")

# %% Descriptores (color, texture, shape) para experimentación
dic_features = {"train": {}, "valid": {}}  # diccionario de features
for folder in ["train", "valid"]:
    route_color = "features_labels_{}_color".format(folder)
    route_texture = "features_labels_{}_texture".format(folder)
    route_shape = "features_labels_{}_shape".format(folder)
    # Cargamos la mejor combinación de cada descriptor (color, texture, shape)
    dic_features[folder] = {"color": scipy.io.loadmat(route_color)["features"],
        "textura": scipy.io.loadmat(route_texture)["features"],
        "forma": scipy.io.loadmat(route_shape)["features"]}
    print("-->Descriptores (color, textura y forma) para {} cargados desde archivos".format(folder))

#%% Experimentacion clasificadores con parametros por defecto
clasificadores = {"SVM": SVC(), "RF": RandomForestClassifier(), "MLP": MLPClassifier()}
df_general = pd.DataFrame(columns=["Clasificador", "Descriptor", "Precisión", "Cobertura", "F1"], index=range(3*3))
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
print("\033[1;36m"+ "Métricas generales experimentación clasificadores automaticos" + '\x1b[0m\n')
print(df_general)
df_general.to_csv("./data_mp3/ResultadosInforme/metricas_general_auto.csv")

#%% Experimento variando parámetros C y gamma de SVM
print("\n\x1b[1;36;47m" + "Máquinas de soporte vectorial" + '\x1b[0m\n')
C = [1e3, 5e3]
gamma = [0.01, 0.001]
feature ="color" # Descriptor con que se obtuvo mejores metricas
features_train = dic_features["train"][feature]
features_valid = dic_features["valid"][feature]
df_general_SVM = pd.DataFrame(columns=["C", "gamma", "Precisión", "Cobertura", "F1"], index=range(len(C) * len(gamma)))
dic_labels_predicted_SVM = {}  # diccionario donde se guardará las predicciones de cada experimento
i = 0
for c in C:
    for g in gamma:
        titulo = "SVM para descriptor de {} con C={} y gamma={}".format(feature, c, g)
        print("\n\x1b[1;36m" + "Experimento: " + titulo + '\x1b[0m\n')
        # --------------ENTRENAMIENTO-----------------
        clasificador = SVC(C=c, gamma=g, kernel='rbf', class_weight='balanced',random_state=42)
        clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
        # ----------------VALIDACIÓN----------------
        predicted_labels = clasificador.predict(features_valid)  # Obtenemos las predicciones(labels) para nuestras imagenes de valid
        dic_labels_predicted_SVM["{}_{}_{}".format(feature,c,g)] = predicted_labels  # Guardamos las predicciones(labels)
        df, precision, f1, recall = Resultados_Cuantitativos(true_labels=dic_labels["valid"],
                                                             predicted_labels=predicted_labels,
                                                             unique_labels=unique_labels, name_experiment=titulo)
        df_general_SVM.iloc[i] = [c, g, precision, recall, f1]
        i += 1

#%% Resultados Cualitativos y Cuantitativos SVM
print("\033[1;36m"+  "Métricas generales de experimentación con SVM variando parámetros C y gamma" + '\x1b[0m\n')
print(df_general_SVM)
df_general_SVM.to_csv("./data_mp3/ResultadosInforme/metricas_general_SVM.csv")

images= np.array(valid) #Creamos el array de imagenes
true_labels = labels_valid #Creamos el array con los labels
feature = "color"; best_c = 5e3; best_gamma = 0.01 #Mejor experiemnto
predicted_labels = dic_labels_predicted_SVM["{}_{}_{}".format(feature,best_c, best_gamma)]
index_clasificadas_bien = true_labels == predicted_labels #Se halla el index de las imagenes bien clasificadas
dic = {"Fortalezas": index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }
columnas = {"Fortalezas": 3,"Debilidades": 1}

for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = columnas[key])
    fig.suptitle("Validación: {} de SVM para descriptor de {} con C={} y gamma={}".format(key, feature, best_c, best_gamma))
    fig.tight_layout()
    fig.savefig("./data_mp3/ResultadosInforme/{}_c{}_gamma{}.png".format(key,best_c, best_gamma))
    fig.show()
    continuar = input("\n\x1b[1;93m"+" Press Enter to continue..."+ '\x1b[0m')

# Guardamos el modelo entrenado del mejor experimento
clasificador = SVC(C=best_c, gamma=best_gamma, kernel='rbf', class_weight='balanced',random_state=42)
clasificador.fit(features_train, dic_labels["train"])
#joblib.dump(clasificador, './data_mp3/Modelos/final_SVM_model_201923972_201923531.pkl')

#%% Experimento variando parámetros n_estimators y max_features para RandomForest
print("\n\x1b[1;36;47m" + "Arboles de decisión" + '\x1b[0m\n')
print("Parametros variando:", "\n-->n_estimators (número de arboles)","\n-->max_features (El número de características a considerar al buscar la mejor división)")
n_trees = [10, 50]
max_feat = ["auto", "log2"]

feature="color" # Descriptor con que se obtuvo mejores metricas
features_train = dic_features["train"][feature]
features_valid = dic_features["valid"][feature]
df_general_RF = pd.DataFrame(columns=["n_trees", "max_features", "Precisión", "Cobertura", "F1"], index=range(len(n_trees) * len(max_feat)))
dic_labels_pred_RF = {}  # diccionario donde se guardará las predicciones de cada experimento
j = 0

for n in n_trees:
    for feat in max_feat:
        titulo = "RandomForest con n_estimators={}, max_features={}".format(n, feat)
        print("\033[1;36m" + "Experimento: " + titulo + '\x1b[0m\n')
        # --------------ENTRENAMIENTO-----------------
        clasificador = RandomForestClassifier(n_estimators=n, max_features=feat, random_state=42)
        clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
        # ----------------VALIDACIÓN----------------
        predicted_labels = clasificador.predict(features_valid)  # Obtenemos las predicciones(labels) para las imagenes de valid
        dic_labels_pred_RF["{}_{}_{}".format(feature, n, feat)] = predicted_labels  # Guardamos las predicciones(labels)
        df, precision, f1, recall = Resultados_Cuantitativos(true_labels=dic_labels["valid"],
                                                             predicted_labels=predicted_labels,
                                                             unique_labels=unique_labels, name_experiment=titulo)
        df_general_RF.iloc[j] = [n, feat, precision, recall, f1]
        j += 1

#%% Resultados Cualitativos y Cuantitativos
print("\033[1;36m"+  "Métricas generales de Experimentación con RandomForest variando parámetros n_estimators y max_features" + '\x1b[0m\n')
print(df_general_RF)
df_general_RF.to_csv("./data_mp3/ResultadosInforme/metricas_general_RF.csv")

images= np.array(valid)  # Creamos el array de imagenes
true_labels = labels_valid  # Creamos el array con los labels
feature = "color"; ntree = 50; maxFeat= "auto"  # Mejor experiemnto
predicted_labels = dic_labels_pred_RF["{}_{}_{}".format(feature, ntree, maxFeat)]
index_clasificadas_bien = true_labels == predicted_labels # Se halla el index de las imagenes bien clasificadas
dic = {"Fortalezas": index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien)}
columnas = {"Fortalezas": 3,"Debilidades":1}
for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels,columns = columnas[key])
    fig.suptitle("Validación: {} de RandomForest con n_estimators={}, max_features={}".format(key, ntree, maxFeat))
    fig.tight_layout()
    fig.savefig("./data_mp3/ResultadosInforme/{}_tree{}_feat{}.png".format(key,ntree, maxFeat))
    fig.show()
    continuar = input("\n\x1b[1;93m"+" Press Enter to continue..."+'\x1b[0m')

# Guardamos el modelo entrenado del mejor experimento
clasificador = RandomForestClassifier(n_estimators=ntree, max_features=maxFeat, random_state=42)
clasificador.fit(features_train, dic_labels["train"])
#joblib.dump(clasificador, './data_mp3/Modelos/final_RF_model_201923972_201923531.pkl')

#%% Experimento variando parámetros n_estimators y max_features para perceptrón multicapa
print("\n\x1b[1;36;47m" + "Perceptrón multicapa" + '\x1b[0m\n')

hidden_layer_sizes = [(50),(50,50)]
activation = ["identity", "logistic"]

feature = "color"
features_train = dic_features["train"][feature]
features_valid = dic_features["valid"][feature]
df_general_MLP = pd.DataFrame(columns=["hidden_layer_sizes","activation", "precision", "recall", "f1"], index=range(len(hidden_layer_sizes)*len(activation)))
dic_labels_p = {}
k = 0

for act in hidden_layer_sizes:
    for lr in activation:
        titulo = "MLP: hidden_layers={}, learning_Rate={}".format(act, lr)
        print("\033[1;36m" + "Experimento: "+ titulo + '\x1b[0m\n')
        # --------------ENTRENAMIENTO-----------------
        clasificador = MLPClassifier(hidden_layer_sizes=act, activation=lr, random_state=42)
        clasificador.fit(features_train, dic_labels["train"])  # Entrenamos a nuestro clasificador
        # ----------------VALIDACIÓN----------------
        predicted_labels = clasificador.predict(features_valid)  # Obtenemos las predicciones(labels) para nuestras imagenes de valid
        dic_labels_p["{}{}{}".format(feature, act, lr)] = predicted_labels  # Guardamos las predicciones(labels)

        df, precision, f1, recall = Resultados_Cuantitativos(true_labels=dic_labels["valid"],predicted_labels=predicted_labels,unique_labels=unique_labels, name_experiment=titulo)
        df_general_MLP.iloc[k] = [act,lr, precision, recall, f1]
        k += 1

#%% Resultados Cualitativos y Cuantitativos de MLP
print("\033[1;36m"+"Métricas generales de experimentación con perceptrón multicapa variando parámetros n_estimators y max_features" + '\x1b[0m\n')
print(df_general_MLP)
df_general_MLP.to_csv("./data_mp3/ResultadosInforme/metricas_general_MLP.csv")

images = np.array(valid)  # Creamos el array de imagenes
true_labels = labels_valid  # Creamos el array con los labels
feature = "color"; hl= (50,50); ac= "identity"  # Mejor experiemnto
predicted_labels = dic_labels_p["{}{}{}".format(feature,hl,ac)]
index_clasificadas_bien = true_labels == predicted_labels # Se halla el index de las imagenes bien clasificadas
dic = {"Fortalezas": index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien)}
columnas = {"Fortalezas": 3,"Debilidades":1}
for key, index in dic.items():
    # Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, images, true_labels, predicted_labels, columns=columnas[key])
    fig.suptitle("Validación: {} de pereceptrón multicapa con hiddenLayers : {} y activation:{}".format(key,hl,ac))
    fig.tight_layout()
    fig.savefig("./data_mp3/ResultadosInforme/{}_hiddenlayer{}_activation{}.png".format(key,hl,ac))
    fig.show()
    continuar = input("\n\x1b[1;93m" + " Press Enter to continue..." + '\x1b[0m')

# Guardamos el modelo entrenado del mejor experimento
clasificador = MLPClassifier(hidden_layer_sizes=hl, activation=ac, random_state=42)
clasificador.fit(features_train, dic_labels["train"])
#joblib.dump(clasificador, './data_mp3/Modelos/final_MLP_model_201923972_201923531.pkl')

#%% Descriptores mixtos
print("\n\x1b[1;36;47m" + "Descriptores mixtos" + '\x1b[0m\n')
dic_mix_features = {"train": {}, "valid": {}}  # diccionario de features
for folder in ["train", "valid"]:
    color = dic_features[folder]["color"]
    textura = dic_features[folder]["textura"]
    forma = dic_features[folder]["forma"]
    # Cargamos la mejor combinación de cada descriptor (color, texture, shape)
    dic_mix_features[folder] = {"color+forma": np.concatenate((color,forma), axis=1),
        "color+textura": np.concatenate((color,textura), axis=1),
        "textura+forma": np.concatenate((forma,textura), axis=1),
        "color+textura+forma": np.concatenate((color,forma,textura), axis=1)}
    print("-->Descriptores mixtos para {} creados".format(folder))

ntree = 50; maxFeat = "auto" #Mejor experimento RF
df_general_mix = pd.DataFrame(columns=["descriptor mixto", "precision", "cobertura", "f-medida"], index=range(4))
dic_labels_predicted_mix = {}
l = 0
for mix_feature in dic_mix_features["train"]:
    titulo = "SVM con descriptor de {} con C={} y gamma={}".format(feature, best_c, best_gamma)
    print("\n\x1b[1;36m" + "Experimento: " + titulo + '\x1b[0m\n')
    # --------------ENTRENAMIENTO-----------------
    features_train = dic_mix_features["train"][mix_feature]
    clasificador = RandomForestClassifier(n_estimators=ntree, max_features=maxFeat, random_state=42)
    clasificador.fit(features_train, dic_labels["train"])
    # ----------------VALIDACIÓN----------------
    features_valid = dic_mix_features["valid"][mix_feature]
    predicted_labels = clasificador.predict(features_valid)  # Obtenemos las predicciones(labels) para nuestras imagenes de valid
    dic_labels_predicted_mix[mix_feature] = predicted_labels  # Guardamos las predicciones(labels)
    df, precision, f1, recall = Resultados_Cuantitativos(true_labels=dic_labels["valid"],
                                                         predicted_labels=predicted_labels,
                                                         unique_labels=unique_labels, name_experiment=titulo)
    df_general_mix.iloc[l] = [mix_feature, precision, recall, f1]
    l += 1
print("\033[1;36m"+  "Métricas generales de experimentación con descriptores mixtos" + '\x1b[0m\n')
print(df_general_mix)
df_general_mix.to_csv("./data_mp3/ResultadosInforme/metricas_general_mix.csv")

#%% Test
true_labels=dic_labels["test"]
route_best_model='final_RF_model_201923972_201923531.pkl'# ruta mejor modelo
print("\033[1;35m"+ "TEST" + '\x1b[0m\n')
if os.path.exists(route_best_model):
    clasificador = joblib.load(route_best_model)  # Cargamos el modelo previamente guardado
    print("Se cargó el mejor clasificador del archivo {}".format(route_best_model))
else:# ------------------------------ENTRENAMIENTO---------------------
    features_train = dic_features["train"]["color"]
    ntree = 50;maxFeat = "auto"  # Mejor experimento RF
    clasificador = RandomForestClassifier(n_estimators=ntree, max_features=maxFeat, random_state=42)
    clasificador.fit(features_train, dic_labels["train"])
# --------------------------------TEST--------------------------
features_test = scipy.io.loadmat("features_labels_test_color")["features"]
predicted_labels = clasificador.predict(features_test) # Obtenemos las predicciones(labels) para nuestras imagenes de valid
#------------Resultados Cualitativos y Cuantitativos---------------
df, precision, f1,  recall= Resultados_Cuantitativos(true_labels=dic_labels["test"], predicted_labels=predicted_labels, unique_labels=unique_labels, name_experiment="TEST")
df.to_csv("./data_mp3/ResultadosInforme/metricas_test.csv")
index_clasificadas_bien = true_labels == predicted_labels #Se halla el index de las imagenes bien clasificadas
dic = {"Fortalezas":index_clasificadas_bien, "Debilidades": np.invert(index_clasificadas_bien) }
columnas = {"Fortalezas": 3,"Debilidades": 1}
for key, index in dic.items():
    #Realizamos una figura con las imagenes indicadas por index
    fig = Resultados_Cualitativos(index, dic_images["test"], dic_labels["test"], predicted_labels,columns = columnas[key])
    fig.suptitle("Test: {} de RF con el descriptor de color".format(key))
    fig.tight_layout()
    fig.savefig("./data_mp3/ResultadosInforme/{}_test.png".format(key))
    fig.show()
    continuar = input("\033[1;36m"+" Press Enter to continue..."+ '\x1b[0m')

print("FIN :D")