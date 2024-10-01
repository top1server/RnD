from __future__ import print_function, division, unicode_literals
import numpy as np

class LinearRegressionClassifier:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = None
    
    def train(self):
        one =  np.ones((self.X.shape[0], 1))
        Xbar = np.concatenate((one, self.X), axis=1)
        A = Xbar.T @ Xbar
        b = Xbar.T @ self.y
        self.w = np.linalg.pinv(A) @ b

    def get_weights(self):
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call the train() method first.")
        return self.w

