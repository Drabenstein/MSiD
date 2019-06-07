import numpy as np

def truncate_images(X):
    X = X.reshape(X.shape[0], 36, 36)
    res = np.zeros((X.shape[0], 28, 28))
    for idx, row in enumerate(X):
        truncated = 0
        while truncated < 8:
            upper = row[0].mean()
            bottom = row[:-1].mean()
            left = row[:][0].mean()
            right = row[:][-1:].mean()
            if upper > bottom:
                row = np.delete(row, [row.shape[0] - 1], axis=1)
            else:
                row = np.delete(row, [0], axis=1)
            if left > right:
                row = np.delete(row, [row.shape[1] - 1], axis=0)
            else:
                row = np.delete(row, [0], axis=0)
            truncated += 1
        res[idx] = row
    return res.reshape(X.shape[0], 784)
