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
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm

# reading in the binary data set
redwine_binary = pd.read_csv('Week 3\\Problem 1\\redwine-binary.csv')
# seperating into our covariates/feratures and our response variable
# can get the response variable by just dropping the `quality` column (which is our response variable)
X = redwine_binary.drop('quality', axis=1)
# now get the response variable by just getting the `quality` column
Y = redwine_binary['quality']
# lets separate it into train and test splits as well
# will use 80% for train, 20% for test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# Check data for scale issues
# fig = plt.figure(figsize=[25, 8])
# ax = fig.add_subplot(1, 1, 1)
# ax.boxplot(X_train.transpose());

# Standardize data
mu = np.mean(X_train)
sigma = np.std(X_train)
X_train = (X_train - mu) / sigma;
X_test = (X_test - mu) / sigma;
# fig = plt.figure(figsize=[25, 8])
# ax = fig.add_subplot(1, 1, 1)
# ax.boxplot(X_train.transpose());
# plt.show()


# helper function made in the lecture examples

# function to do our eval for us, this is quite simple and will
# - create a figure
# - draw a confusion matrix for the trainign data in a sub-fig on the left
# - draw a confusion matrix for the testing data in a sub-fig on the right
# - compute the overall classification accuracy on the testing data
# this has simply been created as we're going to do this for each test that we run
def eval_model(model, X_train, Y_train, X_test, Y_test, label = ''):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 3, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    conf.ax_.set_title('Training Set Performance - {}'.format(label));
    ax = fig.add_subplot(1, 3, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
    conf.ax_.set_title('Test Set Performance - {}'.format(label));
    pred = model.predict(X_test)
    print('Test Accuracy: ' + str(sum(pred == Y_test)/len(Y_test)))

# -------------------------------------------------------------------------------------------
# SVM
# -------------------------------------------------------------------------------------------

print('\nBasic SVM')
svm = SVC()
svm.fit(X_train, Y_train)
eval_model(svm, X_train, Y_train, X_test, Y_test)

plt.show()


# Changing C
# Fails to converge for inf and 1000
# print('\nChanging C')
# for C in [np.inf, 1000, 10, 0.1, 0.001]:
#   print('Linear SVM, with C = {}'.format(C))
#   svm = SVC(C=C, kernel='linear', max_iter=10e5)
#   svm.fit(X_train, Y_train)
#   eval_model(svm, X_train, Y_train, X_test, Y_test, 'C = {}'.format(C))



# RBF Kernel
# print('\nChanging RBF Kernel scale')
# for kernel_scale in [10, 100, 0.1, 'scale']:
#   print('RBF SVM, with scale = {}'.format(kernel_scale))
#   svm = SVC(kernel='rbf', gamma=kernel_scale, max_iter=10e6)
#   svm.fit(X_train, Y_train)
#   eval_model(svm, X_train, Y_train, X_test, Y_test, 'RBF SVM, with scale = {}'.format(kernel_scale))


# Polynomial Kernel
# print('\nChanging Polynomial Kernel degree')
# for degree in [2, 3, 5]:
#   print('Polynomial Kernel SVM, with degree = {}'.format(degree))
#   svm = SVC(kernel='poly', degree=degree, max_iter=10e6)
#   svm.fit(X_train, Y_train)
#   eval_model(svm, X_train, Y_train, X_test, Y_test, 'Polynomial Kernel SVM, with degree = {}'.format(degree))


# RBF with different C
# print('\nChanging RBF Kernel scale AND C')
# for kernel_scale in [10, 100, 0.1, 'scale']:
#   for C in [10, 0.1, 0.001]:
#     print('RBF SVM, with C = {} and scale = {}'.format(C, kernel_scale))
#     svm = SVC(C=C, kernel='linear', max_iter=10e5)
#     svm.fit(X_train, Y_train)
#     eval_model(svm, X_train, Y_train, X_test, Y_test, 'RBF SVM, with C = {} and scale = {}'.format(C, kernel_scale))  


# We can see these aren't as good as our RBF kernels. Keep in mind though that we haven't considered different values 
# of C here (or with the RBF kernels either), so there may exist a better solution that we haven't found. 
# We could quite easily use a nested loop to search through multiple kernel parameters and values of C, and 
# this would actually give us a grid search. This would be something worth exploring in your own time.

# -------------------------------------------------------------------------------------------
# CKNN
# -------------------------------------------------------------------------------------------

print('\nBasic CKNN')
cknn = KNeighborsClassifier(n_neighbors=20)
cknn.fit(X_train, Y_train)
eval_model(cknn, X_train, Y_train, X_test, Y_test)

values_of_k = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# print('\Changing values of k')
# for k in values_of_k:
#     cknn = KNeighborsClassifier(n_neighbors=k)
#     cknn.fit(X_train, Y_train)
#     eval_model(cknn, X_train, Y_train, X_test, Y_test)

# print('\nChanging values of k using Euclidean distance weights')
# for k in values_of_k:
#     cknn = KNeighborsClassifier(n_neighbors=k, weights='distance')
#     cknn.fit(X_train, Y_train)
#     eval_model(cknn, X_train, Y_train, X_test, Y_test)

improved_values_of_k = list(range(64, 74))

# print('\nChanging values of k using manhattan distance weights')
# for k in improved_values_of_k:
#     cknn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
#     cknn.fit(X_train, Y_train)
#     eval_model(cknn, X_train, Y_train, X_test, Y_test)



# -------------------------------------------------------------------------------------------
# Random forests
# -------------------------------------------------------------------------------------------

print('\nBasic Random Forest')
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X_train, Y_train)
eval_model(rf, X_train, Y_train, X_test, Y_test)


print('\nChanging Random Forest depth')
depth = [1, 2, 4, 8, 16, 32, 64, 128, 256]
for d in depth:
    rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=0).fit(X_train, Y_train)
    eval_model(rf, X_train, Y_train, X_test, Y_test)



# plt.show()