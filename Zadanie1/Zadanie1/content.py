# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: ciag wejsciowy Nx1
    :param y: ciag wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: blad sredniokwadratowy pomiedzy wyjsciami y
    oraz wyjsciami uzyskanymi z wielowamiu o parametrach w dla wejsc x
    '''
    return np.square(y - polynomial(x, w)).mean()

def design_matrix(x_train, M):
    '''
    :param x_train: ciag treningowy Nx1
    :param M: stopien wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzedu M
    '''
    rows = np.zeros((x_train.shape[0], M + 1))
    for i in range(x_train.shape[0]):
        rows[i] = [x_train[i] ** j for j in range(M + 1)]
    designMatrix = [row for row in rows]
    return designMatrix

def least_squares(x_train, y_train, M):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu, a err blad sredniokwadratowy
    dopasowania
    '''
    designMatrix = np.array(design_matrix(x_train, M))
    w = np.linalg.inv(designMatrix.transpose() @ designMatrix) @ designMatrix.transpose() @ y_train
    return (w, mean_squared_error(x_train, y_train, w))

def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotke (w,err), gdzie w sa parametrami dopasowanego wielomianu zgodnie z kryterium z regularyzacja l2,
    a err blad sredniokwadratowy dopasowania
    '''
    designMatrix = np.array(design_matrix(x_train, M))
    dmTdm = designMatrix.transpose() @ designMatrix
    w = np.linalg.inv(dmTdm + regularization_lambda * np.identity(dmTdm.shape[0])) @ designMatrix.transpose() @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)

def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, ktore maja byc sprawdzone
    :return: funkcja zwraca krotke (w,train_err,val_err), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym, train_err i val_err to bledy na sredniokwadratowe na ciagach treningowym
    i walidacyjnym
    '''
    ws = []
    trainErrors = []
    valErrors = []
    for M in M_values:
        (w, err) = least_squares(x_train, y_train, M)
        ws.append(w)
        trainErrors.append(err)
        valErrors.append(mean_squared_error(x_val, y_val, w))
    bestWIndex = np.argsort(valErrors)[0]
    return (ws[bestWIndex], trainErrors[bestWIndex], valErrors[bestWIndex])

def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: ciag treningowy wejscia Nx1
    :param y_train: ciag treningowy wyjscia Nx1
    :param x_val: ciag walidacyjny wejscia Nx1
    :param y_val: ciag walidacyjny wyjscia Nx1
    :param M: stopien wielomianu
    :param lambda_values: lista ze wartosciami roznych parametrow regularyzacji
    :return: funkcja zwraca krotke (w,train_err,val_err,regularization_lambda), gdzie w sa parametrami modelu, ktory najlepiej generalizuje dane,
    tj. daje najmniejszy blad na ciagu walidacyjnym. Wielomian dopasowany jest wg kryterium z regularyzacja. train_err i val_err to
    bledy na sredniokwadratowe na ciagach treningowym i walidacyjnym. regularization_lambda to najlepsza wartosc parametru regularyzacji
    '''
    ws = []
    trainErrors = []
    for lambdaVal in lambda_values:
        w, err = regularized_least_squares(x_train, y_train, M, lambdaVal)
        ws.append(w)
        trainErrors.append(err)
    bestWIndex = 0
    bestValError = mean_squared_error(x_val, y_val, ws[bestWIndex])
    for idx, w in enumerate(ws):
        valErr = mean_squared_error(x_val, y_val, w)
        if(valErr < bestValError):
            bestValError = valErr
            bestWIndex = idx
    return (ws[bestWIndex], trainErrors[bestWIndex], bestValError, lambda_values[bestWIndex])