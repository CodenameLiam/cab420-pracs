# import all the important packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


# reading in the multiclass data set
redwine_multiclass = pd.read_csv('Week 3\\Problem 2\\redwine-multiclass.csv')
# seperating into our covariates/feratures and our response variable
# can get the response variable by just dropping the `quality` column (which is our response variable)
X = redwine_multiclass.drop('quality', axis=1)
# now get the response variable by just getting the `quality` column
Y = redwine_multiclass['quality']
# lets see what happends when we split the data manually
X_train = X.iloc[0:1200, :]
X_test = X.iloc[1200:, :]
Y_train = Y.iloc[0:1200]
Y_test = Y.iloc[1200:]

fig = plt.figure(figsize=[25, 8])
ax = fig.add_subplot(1, 2, 1)
ax.boxplot(X_train.transpose());
ax.set_title('Raw Data')

# standardise data - this is the same data as in Q1 with different labels, so we have the same need to standardise
mu = np.mean(X_train)
sigma = np.std(X_train)
X_train = (X_train - mu) / sigma;
X_test = (X_test - mu) / sigma;

ax = fig.add_subplot(1, 2, 2)
ax.boxplot(X_train.transpose());
ax.set_title('Data after standardisation');



# having a look at class imbalance
fig = plt.figure(figsize=[25, 10])
ax = fig.add_subplot(1, 3, 1)
ax.hist(Y, 6)
ax.set_title('Total data set')
ax = fig.add_subplot(1, 3, 2)
ax.hist(Y_train, 6)
ax.set_title('Training data set')
ax = fig.add_subplot(1, 3, 3)
ax.hist(Y_test, 6)
ax.set_title('Test data set')
# There is a significant class imbalance, we are lacking low and high grain ratings
# We do however have at least one of each class in both training and validation sets


# function to do our eval for us, this is quite simple and will
# - create a figure
# - draw a confusion matrix for the trainign data in a sub-fig on the left
# - draw a confusion matrix for the testing data in a sub-fig on the right
# - compute the overall classification accuracy on the testing data
# this has simply been created as we're going to do this for each test that we run
def eval_model(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 2, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    conf.ax_.set_title('Training Set Performance');
    ax = fig.add_subplot(1, 2, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
    conf.ax_.set_title('Test Set Performance');
    pred = model.predict(X_test)
    print('Test Accuracy: ' + str(sum(pred == Y_test)/len(Y_test)))


# -------------------------------------------------------------------------------------------
# Random forests
# -------------------------------------------------------------------------------------------
# print('\n Basic Random Forest')
# rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0).fit(X_train, Y_train)
# eval_model(rf, X_train, Y_train, X_test, Y_test)

# print('\n Deep Random Forest')
# rf = RandomForestClassifier(n_estimators=100, max_depth=32, random_state=0).fit(X_train, Y_train)
# eval_model(rf, X_train, Y_train, X_test, Y_test)

# print('\n Balanced Random Forest')
# rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0, class_weight='balanced_subsample').fit(X_train, Y_train)
# eval_model(rf, X_train, Y_train, X_test, Y_test)

# print('\n Balanced Random Forest with more depth')
# rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0, class_weight='balanced_subsample').fit(X_train, Y_train)
# eval_model(rf, X_train, Y_train, X_test, Y_test)


# deeper tree has massively overfit
# balanced tree has greatly improved performance on the rare classes, at the cost of the common ones
# As we go deeper, we can improve the test accuracy, but at the cost of the rare classes which we again start to lose.



# -------------------------------------------------------------------------------------------
# SVMs
# -------------------------------------------------------------------------------------------

# print('\n 1v1 SVM')
# svm = SVC()
# svm.fit(X_train, Y_train)
# eval_model(svm, X_train, Y_train, X_test, Y_test)

# print('\n Balanced 1v1 SVM')
# svm = SVC(class_weight='balanced')
# svm.fit(X_train, Y_train)
# eval_model(svm, X_train, Y_train, X_test, Y_test)

# print('\n Balanced 1vA SVM')
# onevsall_svm = OneVsRestClassifier(SVC(class_weight='balanced'))
# onevsall_svm.fit(X_train, Y_train)
# eval_model(onevsall_svm, X_train, Y_train, X_test, Y_test)



# param_grid = [
#   {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
#   {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
#   {'C': [0.1, 1, 10, 100], 'degree': [3, 4, 5], 'kernel': ['poly']},
#  ]
# svm = SVC(class_weight='balanced')
# grid_search = GridSearchCV(svm, param_grid)
# grid_search.fit(X_train, Y_train)
# grid_search.cv_results_

# print('\n Best SVM')
# best_system = np.argmin(grid_search.cv_results_['rank_test_score'])
# params = grid_search.cv_results_['params'][best_system]
# print(params)
# svm = SVC().set_params(**params)
# svm.fit(X_train, Y_train)
# eval_model(svm, X_train, Y_train, X_test, Y_test)


# -------------------------------------------------------------------------------------------
# CKNN
# -------------------------------------------------------------------------------------------

print('\n Basic CKNN')
cknn = KNeighborsClassifier(n_neighbors=20)
cknn.fit(X_train, Y_train)
eval_model(cknn, X_train, Y_train, X_test, Y_test)

print('\n Less neighbours CKNN')
cknn = KNeighborsClassifier(n_neighbors=2)
cknn.fit(X_train, Y_train)
eval_model(cknn, X_train, Y_train, X_test, Y_test)


plt.show()