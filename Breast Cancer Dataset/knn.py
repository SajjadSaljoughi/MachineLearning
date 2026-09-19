import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self,k):
        self.Y_train = None
        self.X_train = None
        self.k = k

    def euclidean_distance(self,x1, x2):
        return np.sqrt(np.sum((x1-x2) ** 2, axis=1))

    #Training
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y

    #Predict
    def predict(self,X):
        Y_pred = []
        for x in X:
            distances = self.euclidean_distance(x,self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.Y_train[k_indices]
            Y_pred.append(np.bincount(k_nearest_labels).argmax())
        return np.array(Y_pred)

    def evaluate(self,X,Y):
        Y_predict = self.predict(X)
        accuracy = np.sum(Y_predict == Y)/len(Y)
        return accuracy

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    Y = iris.target
    print(X.shape)
    print(Y.shape)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    knn = KNN(3)
    knn.fit(X,Y)
    accuracy = knn.evaluate(X_test,Y_test)
    print(accuracy)
    knn_sklearn = KNeighborsClassifier(n_neighbors=3)
    knn_sklearn.fit(X_train,Y_train)
    accuracy_sklearn = knn_sklearn.score(X_test,Y_test)
    print(accuracy_sklearn)