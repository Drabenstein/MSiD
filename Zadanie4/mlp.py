import numpy as np
import pickle as pkl
from sklearn.neural_network import MLPClassifier
from file_operations import save_data, read_train_file
from utilities import classification_error

def predict_with_model_params(x_features, W, b, classes):
    def predict_for_single(xi):
        steps = len(W) - 1
        for i in range(steps):
            xi = relu(W[i].T @ xi + b[i])
        xi = W[steps].T @ xi + b[steps]
        return classes[np.argmax(xi)]

    result = list(map(lambda xi: [predict_for_single(xi)], x_features))
    return np.array(result)

def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X

def learn(x_train, y_train, x_val, y_val):
    model = MLPClassifier(solver='sgd', hidden_layer_sizes=100, max_iter=1, warm_start=True, batch_size=40)
    record = 0.8
    record_index = 0
    for i in range(1, 1000):
        model.fit(x_train, y_train)
        W, b, classes = model.coefs_, model.intercepts_, model.classes_
        y_predicted = predict_with_model_params(x_val, W, b, classes)
        result = classification_error(y_predicted, y_val)
        if result > record:
            record = result
            record_index = i
            save_data(
                {
                    'W': W,
                    'b': b,
                    'classes': classes
                }, 'models/mlp_by_hand/' + str(i) + '_' + str(result) + '.pkl')
            print('\n', i, result, sep='---')
        else:
            print(i, result, sep='_', end=" ", flush=True)
    print(record_index, '=', record)

input = read_train_file()
x = input[0]
y = input[1]
splitPoint = int(0.8 * len(x))
x_train = x[:splitPoint]
x_val = x[splitPoint:]
y_train = y[:splitPoint]
y_val = y[splitPoint:].reshape(x_val.shape[0], 1)
learn(x_train, y_train, x_val, y_val)