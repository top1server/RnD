from __future__ import print_function, division, unicode_literals
import numpy as np

class LogisticRegression:
    def __init__(self, X, y):
        """
        Initialize Logistic Regression Algorithm with parameters:
        - X (numpy.ndarray): Input data matrix of shape (d, N)
        - y (numpy.ndarray): Input label vector (0 or 1) of shape (N,)
        - w (numpy.ndarray): Output weight vector [w0 (bias), w1, ...] of shape (d,)
        """
        self.X = X
        self.y = y
        self.w = None
    
    def sigmoid(self, s):
        """
        Activation function using the sigmoid.
        """
        return 1 / (1 + np.exp(-s))
    
    def train(self, w_init, eta, tol=1e-4, max_count=10000):
        """
        Train the logistic regression model using stochastic gradient descent.

        Parameters:
        - w_init (array-like): Initial weights, including bias.
        - eta (float): Learning rate.
        - tol (float): Tolerance for stopping criteria.
        - max_count (int): Maximum number of weight updates.

        Returns:
        - w (numpy.ndarray): Trained weight vector.
        """
        w = [np.array(w_init, dtype=np.float64)]    
        count = 0
        N = self.X.shape[1]
        d = self.X.shape[0]
        check_w_after = 20

        while count < max_count:
            mix_id = np.random.permutation(N)
            for i in mix_id:
                xi = self.X[:, i]
                yi = self.y[i]
                zi = self.sigmoid(np.dot(w[-1], xi))
                w_new = w[-1] + eta * (yi - zi) * xi
                count += 1

                if count % check_w_after == 0 and len(w) >= check_w_after + 1:
                    if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                        self.w = w_new
                        return self.w

                w.append(w_new)

                if count >= max_count:
                    break

        self.w = w[-1]
        
    def get_weights(self):
        if self.w is None:
            raise ValueError("Model has not been trained yet. Call the train() method first.")
        return self.w

    def predict_proba(self, X):
        """
        Predict probability estimates for input data X.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (d, N)

        Returns:
        - probs (numpy.ndarray): Probability estimates of shape (N,)
        """
        return self.sigmoid(np.dot(self.w, X))

    def predict(self, X, threshold=0.5):
        """
        Predict binary labels for input data X.

        Parameters:
        - X (numpy.ndarray): Input data matrix of shape (d, N)
        - threshold (float): Threshold for classification

        Returns:
        - predictions (numpy.ndarray): Predicted labels (0 or 1) of shape (N,)
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

