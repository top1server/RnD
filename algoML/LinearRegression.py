from __future__ import print_function, division, unicode_literals
import numpy as np

class LinearRegression:
    def __init__(self, X, y):
        """
        Init Linear Regression Algorithm with parameters:
        - X(matrix): input data matrix
        - y(vector): input data vector
        - w(vector): output weight vector [w0 (bias), w1, ...]
        """
        self.X = X
        self.y = y
        self.w = None
    
    def train(self):
        """
        Train method:
        Cost func = 1/2 *||y - Xbar.w||**2
        w = (Xbar.T @ Xbar)**-1 @ (Xbar.T @ y)
        """
        one =  np.ones((self.X.shape[0], 1))
        Xbar = np.concatenate((one, self.X), axis=1)
        A = Xbar.T @ Xbar
        b = Xbar.T @ self.y
        self.w = np.linalg.pinv(A) @ b

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

        one = np.ones((X_new.shape[0], 1))
        Xbar_new = np.concatenate((one, X_new), axis=1)
        y_pred = Xbar_new @ self.w
        return y_pred


