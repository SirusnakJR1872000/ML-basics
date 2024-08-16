# Performing Cross-Validation on a Given Model and Dataset

# load the necessary libraries

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# we will load the dataset
iris = load_iris()
x, y = iris.data, iris.target

# now we initialize a model using LR
model = LogisticRegression(max_iter = 100)

# perform k-fold cross validation
scores = cross_val_score(model, x, y, cv = 5, scoring = 'accuracy')

