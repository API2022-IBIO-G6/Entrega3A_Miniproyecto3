from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np

class kmeans_classifier:
    def __init__(self, k=20):
        '''
        :param k: number of clusters to be used.
        '''
        self.model = KMeans(n_clusters=k, random_state=0)
        self.correspondance_dictionary = dict(zip(range(k), -np.ones(k)))


    def fit(self, X, y):
        '''
        :param X: array of input data containing the training vectors. This must be NxM where N is the number
                    of data points and M is the number of features that will be used.
        :param y: target labels to train.
        :return: self (Trained descriptor)
        '''
        # TODO: Realizar el procedimiento para entrenar el modelo.
        self.model.fit(X) #entrenamos al modelo
        clusters = self.model.predict(X) #definimos los clusters asociados a cada descriptor en X
        # -----Se determina la etiqueta de mayor frecuencia para cada clsuter-----
        for i in range(self.model.n_clusters):
            # Se obtiene las etiquetas del cluster i
            labels_by_i = y[clusters == i]
            # Se obtiene cada etiqueta única y la frecuencia de esa etiqueta
            values, counts = np.unique(labels_by_i, return_counts=True)
            # Hallamos la etiqueta con la mayor frecuencia
            largest_label = values[np.argmax(counts)]
            # asociaciamos cada cluster con la clase/etiqueta
            self.correspondance_dictionary[i] = largest_label
        return self

    def predict(self, X):
        '''
        :param X: Array of input data containing the vectors to ve predicted. This must be NxM where N is the number
                    of data points and M is the number of features that will be used.
        :return: Array containing class labels
        '''
        # TODO: Realizar el procedimiento para realizar la descripción de los inputs.
        # llamamos al método predict para hallar los cluster asociados a cada descriptor en X
        predicted_clusters = self.model.predict(X)
        #Creamos el arreglo donde vamos a guardar los labels
        predicted_labels = np.array(["tipo_de_flor"]*len(X))
        for cluster_i, label in self.correspondance_dictionary.items():
            # Se obtiene las posiciones de cada data asignada al cluster i
            index = predicted_clusters == cluster_i
            # Se asigna el label correspondiente al cluster_i
            predicted_labels[index] = label
        return predicted_labels

# Functions for color histograms

def CatColorHistogram(img, num_bins, min_val=None, max_val=None):
    """
    Calculate concatenated histogram for color images
    By: Natalia Valderrama built on Maria Fernanda Roa's code

    Arguments: img (numpy.array) -- 2D color image
    num_bins (array like of ints) -- Number of bins per channel.
    If an int is given, all channels will have same amount of bins.

    Keyword Arguments:
    min_val (array like of ints) -- Minimum intensity range value per channel
    If an int is given, all channels will have same minimum. (default: {None})
    max_val (array like of ints) -- Maximum intensity range value per channel
    If an int is given, all channels will have same maximum. (default: {None})

    Returns: [numpy.array] -- Array containing concatenated color histogram of size num_bins*3.
    """
    assert len(img.shape) == 3, 'img must be a color 2D image'

    # Transform image to float dtype
    img = img_as_float(img)
    _, _, n_channels = img.shape

    # Verify input parameters
    assert isinstance(num_bins, (int, tuple, list, np.array)), 'num_bins must be int or array like'

    if isinstance(num_bins, int):
        num_bins = np.array([num_bins] * n_channels)
    else:
        num_bins = np.array(num_bins)

    assert len(num_bins) == n_channels, 'num_bins length and number of channels differ'

    if min_val is None:
        min_val = np.min(img, (0, 1))
    else:
        assert isinstance(min_val, (int, tuple, list, np.array)), 'min_val must be int or array like'
        if isinstance(min_val, int):
            min_val = np.array([min_val] * n_channels)
        else:
            min_val = np.array(min_val)

    assert len(min_val) == n_channels, 'min_val length and number of channels differ'

    min_val = min_val.reshape((1, 1, -1))

    if max_val is None:
        max_val = np.max(img, (0, 1))
    else:
        assert isinstance(max_val, (int, tuple, list, np.array)), 'max_val must be int or array like'
        if isinstance(max_val, int):
            max_val = np.array([max_val] * n_channels)
        else:
            max_val = np.array(max_val)

    assert len(max_val) == n_channels, 'max_val length and number of channels differ'
    max_val = max_val.reshape((1, 1, -1)) + 1e-5
    concat_hist = np.zeros(np.sum(num_bins), dtype=np.int)
    # Scale intensities (intensities are scaled within the range for each channel)
    # Values now are between 0 and 1
    img = (img - min_val) / (max_val - min_val)
    sum_value = 0

    for c in range(n_channels):
        # Calculate index matrix for each channel

        idx_matrix = np.floor(img[..., c] * num_bins[c]).astype('int')
        idx_matrix = idx_matrix.flatten() + sum_value
        sum_value += num_bins[c]

        # Create concatenated histogram
        for p in range(len(idx_matrix)):
            concat_hist[idx_matrix[p]] += 1

    return concat_hist / np.sum(concat_hist)


def JointColorHistogram(img, num_bins, min_val=None, max_val=None):
    """
    Calculate joint histogram for color images
    By: Maria Fernanda Roa

    Arguments: img (numpy.array) -- 2D color image
    num_bins (array like of ints) -- Number of bins per channel.
    If an int is given, all channels will have same amount of bins.

    Keyword Arguments:
    min_val (array like of ints) -- Minimum intensity range value per channel
    If an int is given, all channels will have same minimum. (default: {None})
    max_val (array like of ints) -- Maximum intensity range value per channel
    If an int is given, all channels will have same maximum. (default: {None})

    Returns: [numpy.array] -- Array containing joint color histogram of size num_bins.
    """

    assert len(img.shape) == 3, 'img must be a color 2D image'

    # Transform image to float dtype
    img = img_as_float(img)
    _, _, n_channels = img.shape

    # Verify input parameters
    assert isinstance(num_bins, (int, tuple, list, np.array)), 'num_bins must be int or array like'

    if isinstance(num_bins, int):
        num_bins = np.array([num_bins] * n_channels)
    else:
        num_bins = np.array(num_bins)

    assert len(num_bins) == n_channels, 'num_bins length and number of channels differ'

    if min_val is None:
        min_val = np.min(img, (0, 1))
    else:
        assert isinstance(min_val, (int, tuple, list, np.array)), 'min_val must be int or array like'
        if isinstance(min_val, int):
            min_val = np.array([min_val] * n_channels)
        else:
            min_val = np.array(min_val)

    assert len(min_val) == n_channels, 'min_val length and number of channels differ'

    min_val = min_val.reshape((1, 1, -1))

    if max_val is None:
        max_val = np.max(img, (0, 1))
    else:
        assert isinstance(max_val, (int, tuple, list, np.array)), 'max_val must be int or array like'
        if isinstance(max_val, int):
            max_val = np.array([max_val] * n_channels)
        else:
            max_val = np.array(max_val)

    assert len(max_val) == n_channels, 'max_val length and number of channels differ'
    max_val = max_val.reshape((1, 1, -1)) + 1e-5

    joint_hist = np.zeros(num_bins, dtype=np.int)
    num_bins = num_bins.reshape((1, 1, -1))

    # Scale intensities (intensities are scaled within the range for each channel)
    # Values now are between 0 and 1
    img = (img - min_val) / (max_val - min_val)

    # Calculate index matrix
    idx_matrix = np.floor(img * num_bins).astype('int')
    idx_matrix = idx_matrix.reshape((-1, n_channels))

    # Create joint histogram
    for p in range(len(idx_matrix)):
        joint_hist[tuple(idx_matrix[p, :])] += 1

    return joint_hist / np.sum(joint_hist)
