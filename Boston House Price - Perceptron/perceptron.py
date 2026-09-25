import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, input_size, lr_w, lr_b, epochs):
        self.w = np.random.rand(input_size, 1)
        self.b = np.random.rand(1, 1)
        self.learning_rate_w = lr_w
        self.learning_rate_b = lr_b
        self.epochs = epochs
        self.losses = []

    def fit(self, X_train, Y_train):
        for j in range(self.epochs):
            for i in range(X_train.shape[0]):
                x = X_train[i]
                y = Y_train[i]

                # print(x.shape)
                # print(self.w.shape)
                y_pred = self.w.T @ x + self.b
                error = y - y_pred

                # SGD
                self.w = self.w + (x.reshape(-1, 1) * error * self.learning_rate_w)
                self.b = self.b + (error * self.learning_rate_b)

                # mae
                loss = np.mean(np.abs(error))
                self.losses.append(loss)

    def predict(self, X_test):
        Y_pred = X_test @ self.w + self.b
        return Y_pred

    def show_plot(self, X_train, Y_train,x_label, y_label,z_label):
        X1 = X_train[:, 0]
        X2 = X_train[:, 1]
        x1 = np.linspace(X1.min(), X1.max(), 50)
        x2 = np.linspace(X2.min(), X2.max(), 50)
        x1, x2 = np.meshgrid(x1, x2)
        eq = self.w[0] * x1 + self.w[1] * x2
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            X1,
            X2,
            Y_train.ravel(),
            color='green'
        )
        ax.plot_surface(x1, x2, eq, alpha=0.5, color='red')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
        plt.show()

    # def evaluate(self, X_test, Y_test,metric):
    #     ...
