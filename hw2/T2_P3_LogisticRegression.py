import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

DEFAULT_ITERS = 10000

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.iters = DEFAULT_ITERS

    def __basis(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def __get_prob(self, x):
        return softmax(np.dot(self.W.T, x))

    def __one_hot(self, val):
        vec = np.zeros(self.n_classes)
        vec[val] = 1
        return vec

    def fit(self, X, y):
        self.X = X
        X = self.__basis(X)
        self.n_classes = max(y) + 1
        # turn y into one-hot vectors
        Y = np.empty(shape=(len(y), self.n_classes), dtype=int)
        for i, val in enumerate(y):
            one_hot_vec = self.__one_hot(val)
            Y[i] = one_hot_vec
        # set vars
        self.y = y
        self.Y = Y
        N = X.shape[0]
        D = X.shape[1]
        self.W = np.random.rand(D, self.n_classes)
        expected_W_shape = self.W.shape
        self.losses = []
        # start gradient descent
        for _ in range(self.iters):
            loss = 0
            grads = [0] * self.n_classes
            for i in range(N):
                # get grad for each class
                for j in range(self.n_classes):
                    prob = self.__get_prob(X[i])[j]
                    loss += Y[i][j] * np.log(prob)
                    grads[j] += ((self.__get_prob(X[i])[j] - Y[i][j]) * X[i]) / N
            # update w_j for each class
            for j in range(self.n_classes):
                grads[j] += self.lam * self.W[:,j]
                self.W[:,j] -= self.eta * grads[j]
                assert self.W.shape == expected_W_shape
            # update loss for plotting iters vs loss
            loss = -loss + .5 * self.lam * np.linalg.norm(self.W)
            self.losses.append(loss)

    def predict(self, X_pred):
        X_pred = self.__basis(X_pred)
        preds = []
        # use calculated weight params to predict X_pred & choose class with max probability
        for x in X_pred:
            prob = self.__get_prob(x)
            indices = np.argmax(prob)
            max_class = indices if np.isscalar(indices) else indices[0]
            preds.append(max_class)
        return np.array(preds)

    # simple loss on self.X and self.Y
    def get_loss(self):
        X = self.__basis(self.X)
        loss = 0
        for i in range(len(self.Y)):
            for j in range(self.n_classes):
                loss += self.Y[i][j] * np.log(self.__get_prob(X[i]))
        loss *= -1
        loss += .5 * self.lam * np.linalg.norm(self.W)
        return loss

    def __print_loss(self, loss=None):
        if loss is None:
            loss = self.get_loss()
        print("Loss (eta =", self.eta, "lam=", self.lam, "iters=", self.iters, ") :", loss)

    # printing loss and plotting for different iters if show_charts = true
    def visualize_loss(self, output_file, show_charts=False):
        self.__print_loss()
        title = "Number of Iters vs. Loss"
        plt.figure()
        plt.title('Number of Iters vs. Loss')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.plot(np.arange(self.iters), self.losses)
        plt.savefig(title + '.png')
        if show_charts:
            plt.show()

    # getting loss for different hyper-params
    def test_hyperparams(self):
        hyperparams = [0.05, 0.01, 0.001]
        for lam in hyperparams:
            for eta in hyperparams:
                self.lam = lam
                self.eta = eta
                self.fit(self.X, self.y)
                self.__print_loss()
