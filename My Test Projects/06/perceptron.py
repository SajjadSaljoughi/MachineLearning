# Perceptron
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("weight-height.csv")
X = data["Height"].values
Y = data["Weight"].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    shuffle=True,
                                                    test_size=0.2)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

fig, (ax1,ax2) = plt.subplots(2, 1)

w = np.random.rand(1, 1)
b = np.random.rand(1, 1)
learning_rate_w = 0.0001
learning_rate_b = 0.5
losses = []

for j in range(100):
    for i in range(X_train.shape[0]):
        x = X_train[i]
        y = Y_train[i]
        y_pred = w * x + b
        error = y - y_pred

        #SGD
        w = w + (error * x * learning_rate_w)
        b = b + (error * learning_rate_b)

        #mae
        loss = np.mean(np.abs(error))
        losses.append(loss)

        # Y_pred = X_train * w + b
        # ax1.clear()
        # ax1.scatter(X_train, Y_train, color="blue")
        # ax1.plot(X_train, Y_pred, color="red")
        #
        # ax2.clear()
        # ax2.plot(losses)
        #
        # plt.pause(0.05)
        # time.sleep(0.1)
Y_pred = X_train * w + b
ax1.scatter(X_train, Y_train, color="blue")
ax1.plot(X_train, Y_pred, color="red")
ax2.plot(losses)
plt.show()