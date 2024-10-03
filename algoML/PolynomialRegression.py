from __future__ import print_function, division, unicode_literals
import numpy as np

class PolynomialRegression:
    def __init__(self, X, y, deg):
        """
        Init Linear Regression Algorithm with parameters:
        - X(matrix): input data matrix
        - y(vector): input data vector
        - w(vector): output weight vector [w0 (bias), w1, ...]
        """
        self.X = X
        self.y = y
        self.deg = deg
        self.w = None
    
    def train(self):
        """
        Train method:
        using vandermonde matrix
        w = (Xbar.T @ Xbar)**-1 @ (Xbar.T @ y)
        """
        Xbar = np.vander(self.x, self.deg+1, increasing=True)
        w = np.linalg.pinv(Xbar.T @ Xbar) @ Xbar.T @ self.y

    def get_weights(self):
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call the train() method first.")
        return self.w
    
    def predict(self, X_new):
        """
        Predict method.
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call the train() method first.")
        
        Xbar_new = np.vander(X_new, self.deg + 1, increasing=True)
        return Xbar_new @ self.w


