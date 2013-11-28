from sklearn.decomposition import ProbabilisticPCA
from sklearn.svm import SVC
from sklearn.decomposition import FastICA
from utils import *
import numpy as np
import pickle

def PCA(X, n_components=150, whiten=True):
    return ProbabilisticPCA(n_components=n_components, whiten=True).fit(X)

def ICA(X, n_components=100):
    return FastICA(n_components=n_components, max_iter=500).fit(X)

def svm_fn(X, y, V=None):
    param_grid = {'C': [1e1,1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    if V is not None:
        clf.fit(X, y)
        pickle.dump(clf, open('classifier.pickle', 'wb'))
        y_pred = clf.predict(V)
        return y_pred
    else:
        print(cross_validation.cross_val_score(clf, X, y, cv=10))

def knn(X, y):
    param_grid = {'n_neighbors': [5]}
    clf = GridSearchCV(KNeighborsClassifier(), param_grid)
    print clf
    print(cross_validation.cross_val_score(clf, X, y, cv=10))

def preprocess(X):
    return X.astype('float64')

def classify(X):
    clf = pickle.load(open('classifier.pickle', 'rb'))
    pca = pickle.load(open('pca.pickle', 'rb'))

    y_pred = clf.predict(pca.transform(X))
    make_submission_file(y_pred, 'test_pred')

if __name__ == '__main__':
    X = load_test()
    classify(X)

