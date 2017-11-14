"""
KNN example from class
"""
import numpy as np  # Llibreria matemÃƒÂ tica
import matplotlib.pyplot as plt  # Per mostrar plots
import sklearn  # Llibreia de DM
import sklearn.datasets as ds  # Per carregar mÃƒÂ©s facilment el dataset digits
import sklearn.model_selection as cv  # Pel Cross-validation
from sklearn.model_selection import learning_curve
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics  # To get more information of the classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif


from sklearn.naive_bayes import GaussianNB  # For numerical featuresm assuming normal distribution
from sklearn.naive_bayes import MultinomialNB  # For features with counting numbers (f.i. hown many times word appears in doc)
from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)

# %matplotlib inline

# Load digits dataset from scikit
digits = ds.load_digits()
# Separate data from labels
X = digits.data
y = digits.target
# Print range of values and dimensions of data
# Data and labels are numpy array, so we can use associated methods
print(('Range: ', X.min(), X.max()))

# Images are 8x8 pixels.
print('Dimension of the dataset:\n Number of samples: ', X.shape[0], '\n Pixel size: ', X.shape[1])
# Just for demostration purposes, let's see some images.
nrows, ncols = 2, 5
plt.figure(figsize=(6, 3))
plt.gray()
for i in range(ncols * nrows):
    ax = plt.subplot(nrows, ncols, i + 1)
    ax.matshow(digits.images[i, ...])
    plt.xticks([]);
    plt.yticks([])
    plt.title(digits.target[i])
# plt.show()

print('Doing CV...')
# Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
(X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1)

# Create a kNN classifier object
knc = nb.KNeighborsClassifier()

print('Training the model...')
# Train the classifier
knc.fit(X_train, y_train)

# Obtain accuracy score of learned classifier on test data
print('Accuracy: ', knc.score(X_test, y_test))

# We calculate de y predicted with the trained model
y_pred = knc.predict(X_test)
print('Our confusion matrix: ')
print(sklearn.metrics.confusion_matrix(y_test, y_pred))

print('Our classification report: ')

print(metrics.classification_report(y_test, y_pred))

one = np.zeros((8, 8))
one[2:-1, 4] = 16  # The image values are in [0, 16].
one[2, 3] = 16
print(one)

# Draw the artificial image we just created
plt.figure(figsize=(2, 2))
plt.imshow(one, interpolation='none')
plt.grid(False)
plt.xticks();
plt.yticks()
plt.title("One")
# plt.show()

# Let's see prediction for the new image
print(knc.predict(one.reshape(1, 64)))

# Method 1
cv_scores = cross_val_score(nb.KNeighborsClassifier(),
                            X=X,
                            y=y,
                            cv=10, scoring='accuracy')

# cv_scores is a list with 10 accuracies (one for each validation)
print(cv_scores)

# Let's get the mean of the 10 validations (and standard deviation of them)
print(np.mean(cv_scores))
print(np.std(cv_scores))

# Method 2
# Build confussion matrix of all 10 cross-validations
predicted = cross_val_predict(nb.KNeighborsClassifier(), X=X, y=y, cv=10)

print(sklearn.metrics.confusion_matrix(y, predicted))
print(sklearn.metrics.accuracy_score(y, predicted))

confmat = sklearn.metrics.confusion_matrix(y, predicted)

fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize=7)

plt.xlabel('Predicted label')
plt.ylabel('True label')

plt.tight_layout()
plt.savefig('ConMatrix.png', dpi=800)
# plt.show()

plt.figure()
print(metrics.classification_report(y_test, y_pred))

train_sizes, train_scores, test_scores = \
    learning_curve(estimator=nb.KNeighborsClassifier(n_neighbors=3),
                   X=X,
                   y=y,
                   train_sizes=np.linspace(0.05, 1.0, 10),
                   cv=10,
                   n_jobs=-1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid(True)
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=600)
# plt.show()

# See parameters in
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# Results with different parameters: k
cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=1), X=X_train, y=y_train, cv=10)
print("Accuracy 1 neighbour:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=3), X=X_train, y=y_train, cv=10)
print("Accuracy 3 neighbours:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=5), X=X_train, y=y_train, cv=10)
print("Accuracy 5 neighbours:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=7), X=X_train, y=y_train, cv=10)
print("Accuracy 7 neighbours:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=9), X=X_train, y=y_train, cv=10)
print("Accuracy 9 neighbours:", np.mean(cv_scores))

# Results with different parameters: k and distance weighting
cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=1, weights='distance'), X=X_train, y=y_train, cv=10)
print("Accuracy 1 neighbour: and distance weighting:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=3, weights='distance'), X=X_train, y=y_train, cv=10)
print("Accuracy 3 neighbour: and distance weighting:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=5, weights='distance'), X=X_train, y=y_train, cv=10)
print("Accuracy 5 neighbour: and distance weighting:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=7, weights='distance'), X=X_train, y=y_train, cv=10)
print("Accuracy 7 neighbour: and distance weighting:", np.mean(cv_scores))

cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=9, weights='distance'), X=X_train, y=y_train, cv=10)
print("Accuracy 9 neighbour: and distance weighting:", np.mean(cv_scores))

plt.figure()
lr = []
for ki in range(1, 30, 2):
    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki), X=X_train, y=y_train, cv=10)
    lr.append(np.mean(cv_scores))
plt.plot(range(1, 30, 2), lr, 'b', label='No weighting')

lr = []
for ki in range(1, 30, 2):
    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki, weights='distance'), X=X_train, y=y_train,
                                cv=10)
    lr.append(np.mean(cv_scores))
plt.plot(range(1, 30, 2), lr, 'r', label='Weighting')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()

# plt.show()

# I look for the best combination of parameters using grid search
params = {'n_neighbors': list(range(1, 30, 2)), 'weights': ('distance', 'uniform')}
knc = nb.KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid=params, cv=10, n_jobs=-1)  # If cv is integer, by default is Stratifyed
clf.fit(X_train, y_train)
print("Best Params=", clf.best_params_, "Accuracy=", clf.best_score_)

# Now I want to apply the new parameters to the test set
parval=clf.best_params_
knc = nb.KNeighborsClassifier(n_neighbors=parval['n_neighbors'],weights=parval['weights'])
knc.fit(X_train, y_train)
pred=knc.predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print(sklearn.metrics.accuracy_score(y_test, pred))

# interval confidence
from statsmodels.stats.proportion import proportion_confint

epsilon = sklearn.metrics.accuracy_score(y_test, pred)
print("Can approximate by Normal Distribution?: ",X_test.shape[0]*epsilon*(1-epsilon)>5)
print("Interval 95% confidence:", "{0:.3f}".format(epsilon), "+/-", "{0:.3f}".format(1.96*np.sqrt(epsilon*(1-epsilon)/X_test.shape[0])))
# or equivalent
proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='normal')

# in the output we see that the distribution isn't normal, because
# of that we change the mode to binomial
print('New confidence interval: ')
print(proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))


# We compare two classifiers ussing the Mcnemar's Test:

# Build two classifiers

# Classifier 1 (3 Neighbours) successes
y_pred = cross_val_predict(nb.KNeighborsClassifier(n_neighbors=3), X=X, y=y,  cv=10)
res1=np.zeros(y.shape)
res1[y_pred==y]=1

# Classifier 2 (7 Neighbours) 2 successes
y_pred = cross_val_predict(nb.KNeighborsClassifier(n_neighbors=7), X=X, y=y,  cv=10)
res2=np.zeros(y.shape)
res2[y_pred==y]=1

# Build contingency matrix
n00 = np.sum([res1[res2==1]==1])
n11 = np.sum([res1[res2==0]==0])
n10 = np.sum([res1[res2==1]==0])
n01 = np.sum([res1[res2==0]==1])

# Chi -square test
print("Have the classifiers significant different accuracy?:",(np.abs(n01-n10)-1)**2/(n01+n10)>3.84)


# We loock at the errors in the test set:


testerrors=[i for i,k in enumerate(pred) if k!=y_test[i]]
plt.gray()
#plt.ion
for i in testerrors:
    plt.matshow(X_test[i].reshape(8,8))
    plt.xticks([]); plt.yticks([])
    print("Guess:", pred[i],"Reality:",y_test[i])
    #plt.show()

"""
Next Exercise: 10 fold CV using Naive Bayes
"""


clf = GaussianNB()
pred = clf.fit(X_train, y_train).predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print()
print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
print()
print(metrics.classification_report(y_test, pred))
epsilon = sklearn.metrics.accuracy_score(y_test, pred)
proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')

# Export data to Rapidminer

import pandas as pd
df = pd.DataFrame(np.c_[ digits.data, digits.target])
df.to_csv("digits2.csv",index=False)

# Go to Rapidminer and load the data set. Reproduce grid Search there and report results on the test set

# Noise stuff:

# Lets' add noise to data: 64 new columns with random data
nrcols=64
col = np.random.randint(0,17,(X_train.data.shape[0],nrcols))

Xr=np.hstack((X_train,col))

col = np.random.randint(0,17,(X_test.data.shape[0],nrcols))
Xr_test=np.hstack((X_test,col))
plt.figure()
lr = []
for ki in range(1,30,2):
    knc = nb.KNeighborsClassifier(n_neighbors=ki)
    knc.fit(X_train, y_train)
    lr.append(knc.score(X_test, y_test))
plt.plot(range(1,30,2),lr,'b',label='No noise')

lr = []
for ki in range(1,30,2):
    knc = nb.KNeighborsClassifier(n_neighbors=ki)
    knc.fit(Xr, y_train)
    lr.append(knc.score(Xr_test, y_test))
plt.plot(range(1,30,2),lr,'r',label='With noise')

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()

plt.show()