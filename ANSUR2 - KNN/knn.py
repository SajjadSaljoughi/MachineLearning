import numpy as np

class KNN:
    def __init__(self,k):
        self.Y_train = None
        self.X_train = None
        self.k = k

    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1-x2) ** 2))

    #Training
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y

    #Predict
    def predict(self,X):
        Y = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                distances.append(self.euclidean_distance(x,x_train))
            nearest_neighbour = np.argsort(distances)[0:self.k]
            y = np.argmax(np.bincount(self.Y_train[nearest_neighbour]))
            Y.append(y)
        return Y

    def evaluate(self,X,Y):
        Y_predict = self.predict(X)
        accuracy = np.sum(Y_predict == Y)/len(Y)
        return accuracy
