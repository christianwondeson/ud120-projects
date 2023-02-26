# let us write our own KNN from scratch
from scipy.spatial import distance

def eub(a, b):
    return distance.euclidean(a, b)

class scrappyKN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
            return predictions

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()


X = iris.data
y = iris.target


