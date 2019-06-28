import numpy as np
from utilities import classification_error
from file_operations import read_model

def predict(X):
    model = read_model('models/sgd/0.001_0.7024444444444444.pkl')
    w = model['w']
    b = model['b']
    pred = X @ w.T + b
    return np.argmax(pred, axis=1).reshape(X.shape[0], 1)
