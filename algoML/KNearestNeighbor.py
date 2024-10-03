from __future__ import print_function, division, unicode_literals
from collections import Counter
import numpy as np

class KNearestNeightbors:
    def __init__(self, K):
        self.K = K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predictAll(self, X_test):
        predictions = [self.predict(x) for x in X_test]
        return np.array(predictions)

    def predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.K]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    

