import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from LLS import LLS

data = pd.read_csv("housing.csv")
data.head()

data.corr()


plt.scatter(data["RM"], data["MEDV"])
plt.show()

plt.scatter(data["LSTAT"], data["MEDV"])
plt.show()

X = data[["RM","LSTAT"]]
Y = data["MEDV"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

lls = LLS()
lls.fit(X_train, Y_train)
w = lls.calculate_slope()


X1 = lls.x_train[:, 0]
X2 = lls.x_train[:, 1]

x1 = np.linspace(X1.min(), X1.max(), 50)
x2 = np.linspace(X2.min(), X2.max(), 50)

x1, x2 = np.meshgrid(x1, x2)

eq = w[0] * x1 + w[1] * x2


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X1,
    X2,
    lls.y_train.ravel(),
    color='green'
)

ax.plot_surface(x1, x2, eq, alpha=0.5,color='red')

ax.set_xlabel('RM')
ax.set_ylabel('LSTAT')
ax.set_zlabel('MEDV')

plt.show()