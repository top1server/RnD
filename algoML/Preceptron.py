from __future__ import print_function, division, unicode_literals
import numpy as np

class Perceptron:
    def __init__(self, X, y):
        """
        Init Preceptron Learning Algorithm with parameters:
        X(matrix): input data, every column is data point(each column has FORM (1, x[i]))
        y(vector): lable input for each data_point
        w(vector): weight vector has FORM: (bias, w[i])
        """
        self.X = X
        self.y = y
        self.w = None

    def train(self, learning_rate=1, max_iter=1000):
        """
        Train method:
        count sum number of misclassified_points
        if predict y[i] != expect y[i] --> w <-- w + y[i] * X[:, i]
        return weight vector and misclassified_points
        """
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
    
    def predict(self, X_new):
        """
        Prefict lable.
        """
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call the train() method first.")
        X_new = np.concatenate([np.ones((1, X_new.shape[1])), X_new], axis=0)
        y_pred = np.sign(np.dot(self.w, X_new))
        return y_pred

