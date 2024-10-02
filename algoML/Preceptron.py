from __future__ import print_function, division, unicode_literals
import numpy as np

class Perceptron:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.w = None
    
    def train(self, learning_rate=1, max_iter=1000):
        X = np.concatenate([np.ones((1, self.X.shape[1])), self.X], axis=0)
        w = np.zeros(X.shape[0])
        misclassified_points = []

        for _ in range(max_iter):
            num_misclassified = 0
            misclassified_points = []
            for i in range(X.shape[1]):
                if np.sign(np.dot(w, X[:, i])) != self.y[i]:
                    w += learning_rate * self.y[i] * X[:, i]
                    num_misclassified += 1
                    misclassified_points.append((X[:, i], self.y[i]))
                
            if num_misclassified == 0:
                print(f"The algorithm converges after {_} iterations.")
                break
        self.w = w
        return w, misclassified_points
    

