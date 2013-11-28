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

# Run random forest method.
def random_forest(X, y, V=None, n_estimators=100):
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=1, random_state=0)
    clf.fit(X, y)
    return clf.predict(V)

# Mixture of Gaussians.
def mog(X, y, V, n_components = 25, cov_type = 'diag'):
    score_train = []
    score_test = []
    score_val = []
    min_score = 0
    
    # Train MoG for each class and compute score.
    classifiers = list(GMM(n_components=n_components, covariance_type=cov_type, n_iter=10) for i in range(1,8))
    for i in range(len(classifiers)):
        classifiers[i].fit(X[y == i + 1])
        score_test.append(classifiers[i].score(V))
        min_score = min(min(score_test[i]), min_score)
    
    # Make prediction.
    y_pred = []
    for j in range(len(X)):
        max_score = min_score
        chosen_class = 0
        for i in range(len(classifiers)):
            if (score_test[i][j] > max_score):
                max_score = score_test[i][j]
                chosen_class = i + 1
        y_pred.append(chosen_class)
        
    return y_pred

# Run SVM, MOG, Random Forest and take majority class.
def my_ensemble(X, y, V, y_test):
    y_pred = []
    votes = []
    
    votes.append(svm_fn(X, y, V))
    votes.append(mog(X, y, V, 10))
    votes.append(random_forest(X, y, V))
    
    for i in range(len(X_test)):
        if votes[1][i] == votes[2][i]:
            y_pred.append(votes[1][i])
        else:
            y_pred.append(votes[0][i])
    
    return y_pred
    
def preprocess_normalize(X):
    for item in X:
        avg_pixel = np.average(item).astype('float64') + 1
        for pixel in item:
            pixel = pixel.astype('float64')/avg_pixel
    return X

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

