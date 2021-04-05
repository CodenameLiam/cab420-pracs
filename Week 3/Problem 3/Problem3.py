import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import tree

cancer = pandas.read_csv('Week 3\\Problem 3\\wdbc.data')
print(cancer.head())

# get the data, we'll just grab 2 dimensions
X = cancer.iloc[:, 2:4].to_numpy()
cancer.iloc[:, 1] = cancer.iloc[:, 1].astype("category")
Y = cancer.iloc[:, 1].cat.codes.to_numpy()

# plot box plot for the data
fig = plt.figure(figsize=[25, 8])
ax = fig.add_subplot(1, 2, 1)
ax.boxplot(X)
ax.set_title('Raw Data')

# standardise data
mu = numpy.mean(X, 0)
sigma = numpy.std(X, 0)
X = (X - mu) / sigma

# box plot after standardisation
ax = fig.add_subplot(1, 2, 2)
ax.boxplot(X)
ax.set_title('Data after standardisation');


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

def eval_model(model, X_train, Y_train, X_test, Y_test):
    fig = plt.figure(figsize=[25, 8])
    ax = fig.add_subplot(1, 3, 1)
    conf = plot_confusion_matrix(model, X_train, Y_train, normalize='true', ax=ax)
    pred = model.predict(X_train)
    conf.ax_.set_title('Training Set Performance: ' + str(sum(pred == Y_train)/len(Y_train)));
    ax = fig.add_subplot(1, 3, 2)
    conf = plot_confusion_matrix(model, X_test, Y_test, normalize='true', ax=ax)
    pred = model.predict(X_test)
    conf.ax_.set_title('Test Set Performance: ' + str(sum(pred == Y_test)/len(Y_test)));
    
    mins = numpy.min(X_train, 0)
    maxs = numpy.max(X_train, 0)
    xx, yy = numpy.meshgrid(numpy.arange(mins[0], maxs[0], 0.025),
                     numpy.arange(mins[1], maxs[1], 0.025))

    Z = model.predict(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = fig.add_subplot(1, 3, 3)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, s=20, edgecolor='k')

svm = SVC(C=1.0, kernel='linear')
svm.fit(X_train, Y_train)
eval_model(svm, X_train, Y_train, X_test, Y_test)

plt.show()