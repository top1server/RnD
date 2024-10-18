import numpy as np
import matplotlib.pyplot as plt
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def compute_loss(y_pred, y_target):
    m = y_target.shape[1]
    loss = -np.sum(y_target * np.log(y_pred + 1e-8)) / m
    return loss

class BasicNN:
    def __init__(self, x_input, y_target):
        self.x_input = x_input
        self.y_target = y_target
        self.weight1 = np.random.randn(3, self.x_input.shape[0])
        self.bias1 = np.zeros((self.weight1.shape[0], 1))
        self.weight2 = np.random.randn(4, self.weight1.shape[0])
        self.bias2 = np.zeros((self.weight2.shape[0], 1))
        self.weight3 = np.random.randn(3, self.weight2.shape[0])
        self.bias3 = np.zeros((self.weight3.shape[0], 1))

    def forward(self):
        self.Z1 = self.weight1 @ self.x_input + self.bias1
        self.A1 = relu(self.Z1)
        self.Z2 = self.weight2 @ self.A1 + self.bias2
        self.A2 = relu(self.Z2)
        self.Z3 = self.weight3 @ self.A2 + self.bias3
        self.y_pred = softmax(self.Z3)
        return self.y_pred

    def backward(self, learning_rate):
        n_samples = self.y_target.shape[1]
        d_Z3 = self.y_pred - self.y_target
        d_weight3 = (1/n_samples) * d_Z3 @ self.A2.T
        d_bias3 = (1/n_samples) * np.sum(d_Z3, axis=1, keepdims=True)
        
        d_A2 = self.weight3.T @ d_Z3
        d_Z2 = d_A2 * relu_derivative(self.Z2)
        d_weight2 = (1/n_samples) * d_Z2 @ self.A1.T
        d_bias2 = (1/n_samples) * np.sum(d_Z2, axis=1, keepdims=True) 

        d_A1 = self.weight2.T @ d_Z2
        d_Z1 = d_A1 * relu_derivative(self.Z1)
        d_weight1 = (1/n_samples) * d_Z1 @ self.x_input.T
        d_bias1 = (1/n_samples) * np.sum(d_Z1, axis=1, keepdims=True)

        self.weight3 -= learning_rate * d_weight3
        self.bias3 -= learning_rate * d_bias3
        self.weight2 -= learning_rate * d_weight2
        self.bias2 -= learning_rate * d_bias2
        self.weight1 -= learning_rate * d_weight1
        self.bias1 -= learning_rate * d_bias1

input_x = np.array([[2, 1, 2, 1], [-1, 1, 1, 2]]).T
y_target = np.array([[0, 0, 1], [0, 1, 0]]).T
num_iters = 100

myCNN = BasicNN(input_x, y_target)
losses = []

for _ in range(num_iters):
    myCNN.forward()
    loss = compute_loss(myCNN.y_pred, y_target)
    losses.append(loss)
    myCNN.backward(0.1)

plt.plot(losses)
plt.title('Loss Curve over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()




    




