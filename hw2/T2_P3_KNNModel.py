import numpy as np

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    def predict(self, X_pred):
        preds = []
        for i, xi in enumerate(X_pred):
            dists = [] # elements will have form (dist, y_val)
            for j, xj in enumerate(self.X):
                dists.append((self.__dist(xi, xj), self.y[j]))
            k_nearest = sorted(dists, key=lambda tup: tup[0])[:self.K]
            ys = [y_val for dist, y_val in k_nearest]
            # predict class with highest count in the nearest neighbors
            pred = np.argmax(np.bincount(ys))
            preds.append(pred)
        return np.array(preds)

    def __dist(self, xi, xj):
        diff = (xi - xj).reshape(xi.shape[0], 1)
        diff[0] /= 3
        return np.dot(diff.T, diff)

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y
