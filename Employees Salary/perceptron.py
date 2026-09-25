import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size,lr_w,lr_b,epochs):
        self.w = np.random.rand(input_size,1)
        self.b = np.random.rand(1,1)
        self.learning_rate_w = lr_w
        self.learning_rate_b = lr_b
        self.epochs = epochs
        self.losses = []

    def fit(self, X_train, Y_train):
        for j in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]
                y_pred = self.w * x + self.b
                error = y - y_pred

                # SGD
                self.w = self.w + (error * x * self.learning_rate_w)
                self.b = self.b + (error * self.learning_rate_b)

                # mae
                loss = np.mean(np.abs(error))
                self.losses.append(loss)

    def predict(self, X_test):
        Y_pred = X_test * self.w + self.b
        return Y_pred

    def show_plot(self,X_train,Y_train,Y_pred):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        Y_pred = X_train * self.w + self.b
        ax1.scatter(X_train, Y_train, color="blue")
        ax1.plot(X_train, Y_pred, color="red")
        ax2.plot(self.losses)
        plt.show()

    # def evaluate(self, X_test, Y_test,metric):
    #     ...
