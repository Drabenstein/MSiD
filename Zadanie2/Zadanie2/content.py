# --------------------------------------------------------------------------
# ------------ Metody Systemowe i Decyzyjne w Informatyce ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A.  Gonczarek, J.  Kaczmar, S.  Zareba, P.  Dąbrowski
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
    X_Array = X.toarray().astype(int)
    X_Array_T = np.transpose(X_train.toarray()).astype(int)
    return np.array(X_Array.shape[1] - X_Array @ X_Array_T - (1 - X_Array) @ (1 - X_Array_T))


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
    def take(indexes, y):
        return np.take(y, indexes)

    sortedEtiquetesIndexes = np.argsort(Dist, 1, 'mergesort')
    return np.apply_along_axis(take, 1, sortedEtiquetesIndexes, y)


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
    y_uniques_count = np.unique(y).shape[0]
    for neighbours in y:
        knn = np.array(neighbours[:k])
        partResult = [np.count_nonzero(knn == i) / k for i in range(y_uniques_count)]
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
    distances = hamming_distance(X_val, X_train)
    sorted_labels = sort_train_labels_knn(distances, y_train)
    bestIndex = 0
    minError = np.inf
    for i in range(np.size(k_values)):
        pyx = p_y_x_knn(sorted_labels, k_values[i])
        error = classification_error(pyx, y_val)
        errors.append(error)
        if minError > error:
            minError = error
            bestIndex = i
    return (minError, k_values[bestIndex], np.array(errors))


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """
    y_count = y_train.shape[0]
    return np.array([np.count_nonzero(y_train == k) / y_count for k in np.unique(y_train)])


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
    X_array = X_train.toarray()
    y_unique_count = np.unique(y_train).shape[0]
    for k in range(y_unique_count):
        y_k_occurances = np.equal(y_train, k)
        y_k_occurances_count = np.sum(y_k_occurances)
        y_k_x_1_matrix = np.apply_along_axis(np.bitwise_and, 0, X_array, y_k_occurances)
        sum_y_k_x_1 = np.sum(y_k_x_1_matrix, 0)
        results.append(np.divide(sum_y_k_x_1 + a - 1, y_k_occurances_count + a + b - 2))
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
    X_array = X.toarray()
    X_array_inverted = ~X_array

    for idx in range(X.shape[0]):
        part1 = p_x_1_y ** X_array[idx]
        part2 = p_x_0_y ** X_array_inverted[idx]
        numerators = np.prod(part1, axis=1) * np.prod(part2, axis=1) * p_y
        denominator = np.sum(numerators)
        results.append([num / denominator for num in numerators])
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
    errors = []
    a_b = []
    for a in a_values:
        partResult = []
        a_array = np.array([a])
        for b in b_values:
            a_b.append((a, b))
            p_y = estimate_a_priori_nb(y_train)
            b_array = np.array([b])
            p_x_1_y = estimate_p_x_y_nb(X_train, y_train, a_array, b_array)
            p_y_x = p_y_x_nb(p_y, p_x_1_y, X_val)
            error = classification_error(p_y_x, y_val)
            partResult.append(error)
        errors.append(partResult)
    errors = np.array(errors)
    bestIndex = np.argsort(errors, axis=None, kind='mergesort')[0]
    bestYIndex = bestIndex // errors.shape[0]
    bestXIndex = bestIndex % errors.shape[1]
    return (errors[bestYIndex, bestXIndex], a_b[bestIndex][0], a_b[bestIndex][1], errors)
