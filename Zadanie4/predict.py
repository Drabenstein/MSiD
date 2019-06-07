import numpy as np
from utilities import classification_error
from file_operations import read_model, read_train_file
import time

def predict(X):
    model = read_model('models/sgd/0.001_0.7023888888888888.pkl')
    w = model['W']
    b = model['b']
    pred = X @ w.T + b
    return np.argmax(pred, axis=1).reshape(X.shape[0], 1)



train_data = read_train_file()
x = train_data[0]

y_pred = predict(x)
y_true = train_data[1]
start = time.perf_counter_ns()
print(classification_error(y_pred, y_true))
end = time.perf_counter_ns()
print('Execution time: ', end-start)

print(y_pred)