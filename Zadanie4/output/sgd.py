import pickle as pkl
import numpy as np
from sklearn.linear_model import SGDClassifier
from file_operations import save_data, read_train_file
from utilities import classification_error

def learn_sgd(x_train, y_train, x_val, y_val):
    best_fit = 0.6
    best_eta = np.inf
    best_iter = 100
    etas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    for eta in etas:
        model = SGDClassifier(loss="log", learning_rate='constant', n_jobs=-1, eta0=eta, max_iter=300)
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_val)
        fit = classification_error(y_predicted, y_val)
        if fit > best_fit:
            best_eta = eta
            best_fit = fit
            save_data(
                {
                    'w': model.coef_,
                    'b': model.intercept_,
                }, 'models/sgd/' + str(eta) + '_' + str(fit) + '.pkl')
        print(eta, fit, sep='~~~~')
    return best_fit, best_eta, best_iter 

input = read_train_file()
splitIndex = int(0.7 * len(input[0]))
x = input[0]
y = input[1]
x_train = x[:splitIndex]
x_val = x[splitIndex:]
y_train = y[:splitIndex]
y_val = y[splitIndex:]

best_fit, best_eta, best_iter = learn_sgd(x_train, y_train, x_val, y_val)

print('\n\nBest fit:\n\n')
print(best_fit, best_eta, best_iter, sep='----')
