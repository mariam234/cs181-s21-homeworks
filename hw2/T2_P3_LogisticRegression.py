import numpy as np
from scipy.special import softmax

ITERS = 10000

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __basis(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def __get_prob(self, x):
        # print("dot", np.dot(self.W.T, x))
        # print("softmax", softmax(np.dot(self.W.T, x)))
        return softmax(np.dot(self.W.T, x))

    def __one_hot(self, val):
        vec = np.zeros(self.n_classes)
        vec[val] = 1
        return vec

    def fit(self, X, y):
        self.X = X
        X = self.__basis(X)
        self.n_classes = max(y) + 1
        Y = np.empty(shape=(len(y), self.n_classes), dtype=int)
        # turn y into one-hot vectors
        for i, val in enumerate(y):
            one_hot_vec = self.__one_hot(val)
            Y[i] = one_hot_vec
        self.Y = Y
        self.W = np.random.rand(X.shape[1], self.n_classes)
        # find optimal weight parameters for each class
        for j in range(self.n_classes):
            w = self.W[:,j]
            expected_w_shape = w.shape
            # run gradient descent
            for _ in range(ITERS):
                grad = 0
                for i in range(X.shape[1]):
                    grad += ((self.__get_prob(X[i])[j] - Y[i][j]) * X[i]) + self.lam + w
                w = w - self.eta * grad
                assert w.shape == expected_w_shape
            self.W[:,j] = w

    def predict(self, X_pred):
        X_pred = self.__basis(X_pred)
        preds = []
        # use calculated weight params to predict X_pred & choose class with max probability
        for x in X_pred:
            prob = self.__get_prob(x)
            indices = np.argmax(prob)
            max_class = indices if np.isscalar(indices) else indices[0]
            preds.append(max_class)
        print("preds", np.array(preds))
        return np.array(preds)

    def visualize_loss(self, output_file, show_charts=False):
        pass
        # Y_preds = self.predict(self.X)
        # for i in len(Y):
        #     for j in self.n_classes:
        #         Y[[i][j] * np.log * Y[[i][j]
        # # need to include regularization?
        # print("Negative log likelihood loss:", sum(self.Y * np.log(Y_preds)))
