# let use begin by using tree train to train our provided dataset

from sklearn import tree
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

X = iris.data
y = iris.target


from sklearn.model_selection import train_test_split
# let us split our testing and training data 
X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size= 0.5)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print(accuracy_score(Y_test, pred))
print("our algorithm prediction : ", pred)
plt.hist([pred], stacked=True, color=["b"])
plt.show()
# the more feature the best data we will have and the for traning data the more accurate prediction are made

features = [[140, 1], [130, 1], [150, 0], [170, 0]] 
# this are the input we use for predictions
labels = [0, 0, 1, 1]
# this are the final outputs 0 for apples and 1 for oranges

# let us create out classifier since we know our data input and output prediction
# classifier is a box of rule to label our input data

clf = tree.DecisionTreeClassifier()

# we need a learing algoritthms observe they overall unique features

clf = clf.fit(features, labels)
result = clf.predict([[150, 1]])

print(result[0])

if(result[0] == 0):
    print("apple")
else:
    print("orange")