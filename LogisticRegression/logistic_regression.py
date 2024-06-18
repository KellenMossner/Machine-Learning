import numpy as np

class LogisticRegression:
    """
    Logistic Regression classifier.
    
    Parameters:
    - lamda (float): Regularization parameter (default: 10e-8).
    - max_iter (int): Maximum number of iterations for optimization (default: 100).
    - tol (float): Tolerance for convergence (default: 1e-10).
    - bias (bool): Whether to include bias term in the model (default: False).
    
    Attributes:
    - lamda (float): Regularization parameter.
    - max_iter (int): Maximum number of iterations for optimization.
    - tol (float): Tolerance for convergence.
    - bias (bool): Whether to include bias term in the model.
    - weights (ndarray): Model weights.
    """
    
    def __init__(self, lamda=10e-8, max_iter=100, tol=1e-10, bias=False):
        self.lamda = lamda
        self.max_iter = max_iter
        self.tol = tol
        self.bias = bias
        self.weights = None
        
    def fit(self, X, y):
        """
        Fit the logistic regression model to the training data.
        
        Parameters:
        - X (ndarray): Training input samples, shape (d, N).
        - y (ndarray): Target values, shape (N,).
        """
        d, N  = X.shape
        if self.bias:
            X = np.vstack((np.ones((1,N)),X))
            w = np.zeros((d+1,1))
        else:
            w = np.zeros((d,1))
        
        for i in range(self.max_iter):
            w_prior = w
            w = w - np.linalg.inv(self._H(w, X)) @ self._l(w, X, y)
            
            if (np.linalg.norm(w_prior-w) < self.tol):
                break
                
        self.weights = w
        
    def predict(self, X):
        """
        Predict the class labels for the input samples.
        
        Parameters:
        - X (ndarray): Input samples, shape (d, N).
        
        Returns:
        - y_pred (ndarray): Predicted class labels, shape (N,).
        """
        if self.bias:
            X = np.vstack((np.ones((1,N)),X))
        w = self.weights
        y = self._sigmoid(w.T @ X)
        y_pred = (y > 0.5).astype(int)
        return y_pred.flatten()
        
    def _sigmoid(self, x):
        """
        Compute the sigmoid function element-wise.
        
        Parameters:
        - x (ndarray): Input values.
        
        Returns:
        - result (ndarray): Sigmoid function values.
        """
        return 1 / (1 + np.exp(-x))
    
    def _H(self, w, X):
        """
        Compute the Hessian matrix for the logistic regression model.
        
        Parameters:
        - w (ndarray): Model weights, shape (d, 1).
        - X (ndarray): Input samples, shape (d, N).
        
        Returns:
        - hessian (ndarray): Hessian matrix, shape (d, d).
        """
        d, N = X.shape
        hessian = np.zeros((d,d))
        I = np.identity(d)
        
        for i in range(N):
            xn = X[:,i].reshape(d, 1)
            hessian += self._sigmoid(w.T @ xn) @ (1 - self._sigmoid(w.T @ xn)) * (xn @ xn.T) + (1/self.lamda) * I
        return hessian
    
    def _l(self, w, X, y):
        """
        Compute the gradient of the log-likelihood function for the logistic regression model.
        
        Parameters:
        - w (ndarray): Model weights, shape (d, 1).
        - X (ndarray): Input samples, shape (d, N).
        - y (ndarray): Target values, shape (N,).
        
        Returns:
        - l (ndarray): Gradient of the log-likelihood function, shape (d, 1).
        """
        d, N = X.shape
        l = np.zeros((d,1))
        
        for i in range(N):
            xn = X[:,i].reshape(d, 1)
            yn = y[i]
            l += (yn - self._sigmoid(w.T @ xn)) * xn + (1/self.lamda) * w
        return -l
    
    def get_params(self):
        """
        Get the model weights.
        
        Returns:
        - weights (ndarray): Model weights, shape (d, 1).
        """
        return self.weights
    
    def get_bias(self):
        """
        Get the bias term status.
        
        Returns:
        - bias (bool): Whether bias term is included in the model.
        """
        return self.bias