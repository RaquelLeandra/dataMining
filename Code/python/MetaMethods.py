"""
This code contains the contents of the Meta Methods laboratory from class.
"""

# Import libraries

import numpy as np  # Numeric and matrix computation
import pandas as pd  # Optional: good package for manipulating data
import sklearn as sk  # Package with learning algorithms implemented

# TODO: no me gusta como importa las librerías, cambiarlo.
from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
df = pd.read_csv(url, header=None)
print(df.head())

# No preprocessing needed. Numerical and scaled data
# Separate data from labels

y = df[34].values
X = df.values[:, 0:34]

"""
Naive Bayes, KNN and Decision trees: 

We do 50 rounds of cv (using GridSearchCV )because the dataset is small enough.
Then we calculate the Naive Bayes classifier, the KNN and the decision Tree.
"""

cv = 50

clf1 = GaussianNB()

params = {'n_neighbors': list(range(1, 30, 2)), 'weights': ('distance', 'uniform')}
knc = KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid=params, cv=cv, n_jobs=-1)  # If cv is integer, by default is Stratifyed
clf.fit(X, y)
print("Best Params fo Knn=", clf.best_params_, '\n', "Accuracy=", clf.best_score_)
parval = clf.best_params_
clf2 = KNeighborsClassifier(n_neighbors=parval['n_neighbors'], weights=parval['weights'])

clf3 = DecisionTreeClassifier(criterion='entropy')
print('Comparative of the methods: ')
for clf, label in zip([clf1, clf2, clf3], ['Naive Bayes', 'Knn (3)', 'Dec. Tree', ]):
    scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.3f [%s]" % (scores.mean(), label))

"""
Majority Voting and Weight Voting: 

Now we do a majority voting and weight voting of the three methods. 
"""

# TODO: mirarme como funciona por dentro.

print('Majority voting: ')
eclf = VotingClassifier(estimators=[('nb', clf1), ('knn3', clf2), ('dt', clf3)], voting='hard')
scores = cross_val_score(eclf, X, y, cv=cv, scoring='accuracy')
print("Accuracy: %0.3f [%s]" % (scores.mean(), "Majority Voting"))

print('Weight voting: ')
eclf = VotingClassifier(estimators=[('nb', clf1), ('knn3', clf2), ('dt', clf3)], voting='soft', weights=[2, 1, 2])
scores = cross_val_score(eclf, X, y, cv=cv, scoring='accuracy')
print("Accuracy: %0.3f [%s]" % (scores.mean(), "Weighted Voting"))

"""
Bagging method: 

It's used to reduce variance of a method. 
Consists in taking random samples and fitting the classifiers on them.

In our case we use it with decission trees. This way we obtain more diversity on the trees.
"""

print('Bagging with decision trees with different rows')
for nest in [1, 2, 5, 10, 20, 50, 100, 200]:
    scores = cross_val_score(BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=nest), X, y, cv=cv,
                             scoring='accuracy')
    print("Accuracy: %0.3f [%s]" % (scores.mean(), nest), ' decision trees')

print('Bagging with decision trees with different rows and different columns (selected randomly)')
for nest in [1, 2, 5, 10, 20, 50, 100, 200]:
    scores = cross_val_score(
        BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=nest, max_features=0.35), X, y, cv=cv,
        scoring='accuracy')
    print("Accuracy: %0.3f [%s]" % (scores.mean(), nest), ' decision trees')

"""
Random forest:

Similar to the bagging but using random forest.
Two different implementations.
The first one selects the best threshold, 
The second one selects it randomly
"""
print('Random forest method one (Best threshold):')
for nest in [1, 2, 5, 10, 20, 50, 100, 200]:
    scores = cross_val_score(RandomForestClassifier(n_estimators=nest), X, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.3f [%s" % (scores.mean(), nest), ' number of trees]')

print('Random forest method two(Random threshold):')
for nest in [1, 2, 5, 10, 20, 50, 100, 200]:
    scores = cross_val_score(ExtraTreesClassifier(n_estimators=nest), X, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.3f [%s" % (scores.mean(), nest), ' number of trees]')

"""
Bosting: 

To make new classifiers we use the errors from the previous classifiers, making a classifier
series. It's based on a decision tree classifier.
"""

for nest in [1, 2, 5, 10, 20, 50, 100, 200]:
    scores = cross_val_score(AdaBoostClassifier(n_estimators=nest), X, y, cv=cv, scoring='accuracy')
    print("Accuracy: %0.3f [%s" % (scores.mean(), nest), ' trees]')

# The same restricting the max deph of the trees. 
for nest in [1, 2, 5, 10, 20, 50, 100, 200]:
    scores = cross_val_score(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), n_estimators=nest), X, y, cv=cv,
                             scoring='accuracy')
    print("Accuracy: %0.3f [%s]" % (scores.mean(), nest))
