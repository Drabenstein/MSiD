import numpy as np

def classification_error(y_predicted, y_true):
    return np.count_nonzero(y_predicted == y_true) / len(y_predicted)