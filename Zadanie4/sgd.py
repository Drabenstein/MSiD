import pickle as pkl
import numpy as np
from sklearn.linear_model import SGDClassifier
from file_operations import save_data, read_train_file
from images_truncate import truncate_images

def learn_sgd(x_train, y_train, x_val, y_val):
    best_fit = 0.6
    best_eta = np.inf
    best_iter = 100
    etas = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    for eta in etas:
        model = SGDClassifier(loss="hinge", alpha=eta, learning_rate='optimal', n_jobs=-1, eta0=eta, max_iter=300, warm_start=True)
        model.fit(x_train, y_train)
        y_predicted = model.predict(x_val)
        fit = np.count_nonzero(y_predicted == y_val) / len(y_val)
        if fit > best_fit:
            best_eta = eta
            best_fit = fit
            save_data(
                {
                    'W': model.coef_,
                    'b': model.intercept_,
                    'classes': model.classes_
                }, 'models/sgd_cropped/' + str(eta) + '_' + str(fit) + '.pkl')
        print(eta, fit, sep='~~~~')
    return best_fit, best_eta, best_iter 

# def compute_feats(image, kernels):
#     feats = np.zeros((len(kernels), 2), dtype=np.double)
#     for k, kernel in enumerate(kernels):
#         filtered = ndi.convolve(image, kernel, mode='wrap')
#         feats[k, 0] = filtered.mean()
#         feats[k, 1] = filtered.var()
#     return feats

# kernels = []
# for theta in range(4):
#     theta = theta / 4. * np.pi
#     for sigma in (1, 3):
#         for frequency in (0.05, 0.25):
#             kernel = np.real(gabor_kernel(frequency, theta=theta,
#                                           sigma_x=sigma, sigma_y=sigma))
#             kernels.append(kernel)


input = read_train_file()
splitIndex = int(0.7 * len(input[0]))
x = input[0]
y = input[1]
x_train = x[:splitIndex]
x_val = x[splitIndex:]
y_train = y[:splitIndex]
y_val = y[splitIndex:]

#print(len(kernels))
#print(x_train[0].shape)
#print(ndi.convolve(x_train[0], kernels, mode='wrap'))

x_train_filtered = x_train
x_val_filtered = x_val
#x_train_filtered = np.array([apply_gabor_filter(xi, filter) for xi in x_train])
#x_val_filtered = np.array([apply_gabor_filter(xi, filter) for xi in x_val])

best_fit, best_eta, best_iter = learn_sgd(x_train_filtered, y_train, x_val_filtered, y_val)

print('\n\nBest fit:\n\n')
print(best_fit, best_eta, best_iter, sep='----')
