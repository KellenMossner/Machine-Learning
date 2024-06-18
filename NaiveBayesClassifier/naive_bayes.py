import numpy as np

class NaiveBayes:
    """
    Naive Bayes classifier implementation.

    Attributes:
        priors (list): List of prior probability functions for each class.
        prior_probs (list): List of prior probabilities for each class.
        y (array-like): Array of target values.

    Methods:
        fit(X, y): Fit the Naive Bayes classifier to the training data.
        predict(X): Predict the class labels for the input data.

    """

    def __init__(self):
        self.priors = []
        self.prior_probs = []
        self.y = []

    def fit(self, X, y):
        """
        Fit the Naive Bayes classifier to the training data.

        Args:
            X (array-like): Training data matrix of shape (n_features, n_samples).
            y (array-like): Target values array of shape (n_samples,).

        Returns:
            None

        """
        self.y = y
        d, N = X.shape
        classes = np.unique(y)
        class_counts = np.bincount(y)

        # Calculate all prior probabilities for each class P(Cj)
        self.prior_probs = class_counts / N

        for c in classes:
            X_c = X[:, y == c]
            mu_c = np.mean(X_c, axis=1).reshape(d, 1)
            sigma_c = np.diag(np.var(X_c, axis=1))

            p = lambda x, mu=mu_c, sigma=sigma_c: ((1 / np.sqrt(np.linalg.det(2 * np.pi * sigma))) *
                                                   np.exp(-0.5 * (x - mu).T @ np.linalg.inv(sigma) @ (x - mu)))
            self.priors.append(p)

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Args:
            X (array-like): Input data matrix of shape (n_features, n_samples).

        Returns:
            array-like: Predicted class labels array of shape (n_samples,).

        """
        d, N = X.shape
        classes = np.unique(self.y)
        C = np.zeros((N))

        for i in range(N):
            p_cj = []
            for c in classes:
                p = self.priors[c]
                p_cj.append(p(X[:, i].reshape(d, 1)) * self.prior_probs[c])
            C[i] += np.argmax(p_cj)
        return C
    
    def evaluate(self, X, y):
        """
        Evaluate the performance of the model.

        Args:
            X (array-like): Input data matrix of shape (n_features, n_samples).
            y (array-like): Target values array of shape (n_samples,).

        Returns:
            float: Accuracy of the model.

        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy