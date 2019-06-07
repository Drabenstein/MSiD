import time
from file_operations import read_train_file
from utilities import classification_error
from predict import predict


train_data = read_train_file()
x = train_data[0]

y_pred = predict(x)
y_true = train_data[1]
start = time.perf_counter_ns()
print(classification_error(y_pred, y_true))
end = time.perf_counter_ns()
print('Execution time: ', end-start)

print(y_pred)