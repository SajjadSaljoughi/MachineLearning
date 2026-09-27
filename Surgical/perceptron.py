import numpy as np
from tqdm import tqdm


class Perceptron:
    def __init__(self, learning_rate, input_length, epochs=100):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(input_length)
        self.bias = np.random.rand(1)
        self.loss_history = []
        self.accuracy_history = []
        self.test_loss_history = []
        self.test_accuracy_history = []
        self.epochs = epochs

    def activation(self, x, function):
        if function == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif function == "relu":
            return np.maximum(0, x)
        elif function == "tanh":
            return np.tanh(x)
        elif function == "linear":
            return x
        elif function == "softmax":
            return np.exp(x) / np.sum(np.exp(x))
        else:
            raise ValueError("Function not recognized")

    def fit(self, X_train, Y_train,
            X_test, Y_test,
            activation_function):
        loss_history = []
        accuracy_history = []
        for epoch in tqdm(range(self.epochs)):
            for x, y in zip(X_train, Y_train):
                # forwarding
                y_pred = x @ self.weights + self.bias
                y_pred = self.activation(y_pred, activation_function)
                # back propagation
                eror = y - y_pred

                # updating
                self.weights += self.learning_rate * eror * x
                self.bias += self.learning_rate * eror

            loss = (self.calculate_loss(X_train, Y_train, "mae"))
            self.loss_history.append(loss)
            accuracy = (self.calculate_accuracy(X_train, Y_train))
            self.accuracy_history.append(accuracy)

            loss_test = (self.calculate_loss(X_test, Y_test, "mae"))
            self.test_loss_history.append(loss_test)
            accuracy_test = (self.calculate_accuracy(X_test, Y_test))
            self.test_accuracy_history.append(accuracy_test)

    def predict(self, X_test , activation_function):
        Y_pred = []
        for x in X_test:
            y_pred = x @ self.weights + self.bias
            y_pred = self.activation(y_pred, activation_function)
            Y_pred.append(y_pred)
        return np.array(Y_pred)

    def calculate_loss(self, X_test, Y_test, metric):
        Y_pred = self.predict(X_test)
        if metric == "mse":
            return np.mean(np.square(Y_test - Y_pred))
        elif metric == "mae":
            return np.mean(np.abs(Y_test - Y_pred))
        elif metric == "rsme":
            return np.sqrt(np.mean(np.sqrt(Y_test - Y_pred)))
        else:
            raise ValueError("Metric not recognized")

    def calculate_accuracy(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        Y_pred = Y_pred.reshape(-1)
        Y_pred = np.where(Y_pred > 0.5, 1, 0)
        accuracy = np.sum(Y_pred == Y_test) / len(Y_pred)
        return accuracy

    def evaluate(self, X_test, Y_test):
        loss = self.calculate_loss(X_test, Y_test, "mae")
        accuracy = self.calculate_accuracy(X_test, Y_test)
        return loss, accuracy
