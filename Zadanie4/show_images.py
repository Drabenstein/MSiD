from file_operations import read_train_file
from matplotlib import pyplot as plt
import numpy as np

data = read_train_file()
x = data[0]
images = x.reshape((x.shape[0], 36, 36))

for index in range(images.shape[0]):
    print(images[index])
    plt.imshow(images[index], cmap='gray')
    plt.waitforbuttonpress()