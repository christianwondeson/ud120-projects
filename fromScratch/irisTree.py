import numpy as np
import graphviz
from sklearn.datasets import load_iris

# know let choose our training algorithm
from sklearn import tree

iris = load_iris()

print(iris.feature_names)
# this are the features we observe from the dataset
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])

# for i in range(len(iris.target)):
#     print("example %d: label %s, features %s" %(i, iris.target[i], iris.data[i]))

# let us train the dataset by training those 3 labels of flower setosa, versicolor, virginica
# the first setosa , first versicolor, first virginica
test_idx = [0, 50, 100]

# let us train removing the label and features from the datset for training
# the majority of the data is our training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# let us test the removed data based on the label
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# let us create a classifier and train it
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
print(test_target)

print(clf.predict(test_data))

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
# graph.render("iris") 

dot_data = tree.export_graphviz(clf, out_file=None, 
feature_names=iris.feature_names,  
class_names=iris.target_names,  
filled=True, rounded=True,  
special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 