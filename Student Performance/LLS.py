import numpy as np


class LLS:
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.a = None

    def fit(self, x_train, y_train):
        self.x_train = np.array(x_train).reshape(-1, 1)
        self.y_train = np.array(y_train).reshape(-1, 1)

    def calculate_slope(self):
        a = np.matmul(np.matmul(np.linalg.inv(np.matmul(self.x_train.T, self.x_train)), self.x_train.T),
                           self.y_train)
        return a[0]

    def predict(self,x):
        self.a = self.calculate_slope()
        y = self.a * x
        return y


