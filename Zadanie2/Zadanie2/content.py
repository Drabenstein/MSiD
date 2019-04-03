# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    result = []
    for xi in X.toarray().astype(int):
        partResult = [np.count_nonzero(xi != xj) for xj in X_train.toarray().astype(int)]
        result.append(partResult)
    return np.array(result)


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    result = []
    for d in Dist:
        sortedEtiquetesIndexes = np.argsort(d, -1, 'mergesort')
        result.append([y[i] for i in sortedEtiquetesIndexes])
    return np.array(result)


def p_y_x_knn(y, k):    
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    results = []
    for neighbours in y:
        knn = np.array(neighbours[:k])
        partResult = []
        for i in range(np.unique(neighbours).shape[0]):
            count = np.count_nonzero(knn == i)
            length = np.size(knn)
            partResult.append(count / length)
        results.append(partResult)
    return np.array(results)


def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    modelErrors = 0
    for idx, y in np.ndenumerate(y_true):
        row = p_y_x[idx[0]]
        rowLen = len(row)
        argmaxIndex = np.argmax(row[::-1])
        modelErrors += 0 if rowLen - argmaxIndex - 1 == y else 1
    return modelErrors / y_true.shape[0]


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    errors = []
    for k in k_values:
        distances = hamming_distance(X_val, X_train)
        sorted = sort_train_labels_knn(distances, y_train)
        pyx = p_y_x_knn(sorted, k)
        error = classification_error(pyx, y_val)
        errors.append(error)
    bestIndex = np.argsort(errors)[0]
    return (errors[bestIndex], k_values[bestIndex], np.array(errors))


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    results = []
    for k in np.unique(y_train):
        results.append(np.count_nonzero(y_train == k) / y_train.shape[0])
    return results


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    results = []
    for y in np.unique(y_train):
        partResult = []
        for horizontalIdx in range(X_train.shape[1]):
            countNum = 0
            for verticalIdx in range(X_train.shape[0]):
                countNum += 1 if y_train[verticalIdx] == y and X_train[verticalIdx, horizontalIdx] == True else 0
            numerator = countNum + a[0] - 1
            countDenom = np.count_nonzero(y_train == y)
            denominator = countDenom + a[0] + b[0] - 2
            partResult.append(numerator / denominator)
        results.append(partResult)
    return np.array(results)
            

def p_y_x_nb(p_y, p_x_1_y, X):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    results = []
    p_x_0_y = 1 - p_x_1_y
    for idx in range(X.shape[0]):
        partResult = []
        for yIdx in range(p_y.shape[0]):
            numerator = p_y[yIdx]
            for xIdx in range(X.shape[1]):
                numerator *= p_x_1_y[yIdx, xIdx] if X[idx, xIdx] == True else p_x_0_y[yIdx, xIdx]
            partResult.append(numerator)
        denominator = np.sum(partResult)
        results.append([part / denominator for part in partResult])
    return np.array(results)


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    pass
