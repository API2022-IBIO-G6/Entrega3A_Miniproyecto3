import csv
import scipy.io
import numpy as np

#%%
print("\n\x1b[1;36;47m" +"bla"+ '\x1b[0m\n')

#%%
def load_features_labels(filepath):
    """Función que carga el csv con features y labels"""
    file = open(filepath, encoding="utf-8") #abrimos el archivo
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

for folder in ["train", "valid"]:
    route = 'data_mp3//color//features_labels_{}_{}_{}_{}.csv'.format(folder, TYPE, colorSpace, spaceBins)

    # Cargamos el descriptor y los labels correspondientes al mejor experimento
    features, labels = load_features_labels(route)
    
    dic_labels_features = {"labels": labels, "features": features}
    scipy.io.savemat(route[:-3]+"mat", dic_labels_features)  # guardamos los labels y features (texture)
    print('-->Archivo \"{}\" creado'.format(route))

