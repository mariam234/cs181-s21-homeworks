import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __one_hot(self, val):
        vec = np.zeros(self.n_classes)
        vec[val] = 1
        return vec

    def fit(self, X, y):
        self.n_classes = max(y) + 1
        # turn y into one-hot vectors
        Y = np.empty(shape=(len(y), self.n_classes), dtype=int)
        for i, val in enumerate(y):
            one_hot_vec = self.__one_hot(val)
            Y[i] = one_hot_vec
        N = X.shape[0]
        D = X.shape[1]
        y_counts = []
        self.priors = []
        self.x_means = []
        cov_sums = []
        self.covs = []
        for k in range(self.n_classes):
            indices = np.where(y == k)
            y_counts.append(len(indices[0]))
            self.priors.append(y_counts[k] / N)
            X_filtered = X[indices]
            Y_filtered = Y[indices]
            # print("x_filt", X_filtered)
            # print("x_filt sum", sum(X_filtered))
            # print("y_filt", Y_filtered)
            self.x_means.append(sum(X_filtered) / y_counts[k])
            cov_sum = 0
            for i in range(len(X_filtered)):
                d = np.reshape(X_filtered[i] - self.x_means[k], (D, 1))
                cov_sum += np.dot(d, d.T)
                # print("d", d)
                # print("dot", np.dot(d, d.T))
                # print("cov_sum", cov_sum)
            # check that cov sum symmetric
            assert np.allclose(cov_sum, cov_sum.T)
            cov_sums.append(cov_sum)
            if not self.is_shared_covariance:
                self.covs.append(cov_sum / y_counts[k])
        if self.is_shared_covariance:
            self.cov = sum(cov_sums) / N
        # print("Y_COUNTS", y_counts)
        # print("PRIORS", priors)
        # print("X_MEANS", x_means)
        # for i, cov in enumerate(covs):
        #     print("COV MATRIX", i, ":", cov)

    def predict(self, X_pred):
        preds = []
        for x in X_pred:
            probs = []
            for k in range(self.n_classes):
                cov = self.cov if self.is_shared_covariance else self.covs[k]
                cond_prob = mvn.pdf(x, mean=self.x_means[k], cov=cov)
                probs.append(cond_prob * self.priors[k])
            preds.append(np.argmax(probs))
            # preds.append(probs.index(max(probs)))
        return np.array(preds)

    def negative_log_likelihood(self, X, y):
        # turn y into one-hot vectors
        Y = np.empty(shape=(len(y), self.n_classes), dtype=int)
        for i, val in enumerate(y):
            one_hot_vec = self.__one_hot(val)
            Y[i] = one_hot_vec
        # get neg log likelihood
        eps = 1e-30
        sum = 0
        for i in range(X.shape[0]):
            for k in range(self.n_classes):
                cov = self.cov if self.is_shared_covariance else self.covs[k]
                cond_prob = mvn.pdf(X[i], mean=self.x_means[k], cov=cov)
                sum += Y[i][k] * (np.log(cond_prob + eps) + np.log(self.priors[k] + eps))
        loss = sum * -1
        return loss
