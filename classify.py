from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.decomposition import RandomizedPCA, ProbabilisticPCA
from sklearn.decomposition import FastICA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from autoencoder import Autoencoder
from sklearn.linear_model import SGDClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.lda import LDA
from sklearn.qda import QDA
from time import time
from utils import *
from mlp import MLPClassifier
from nn import TrainNN
import numpy as np

rng = np.random.RandomState(0)
divide = np.vectorize(lambda x: x/255.0)

def label_spreading(X_train, y_train, Xunlabelled, X_test, y_test):
    #pca = randomized_PCA(X_train)
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr_images, tr_labels, test_size=0.3)
    #X = pca.transform(X)
    #val_images = pca.transform(val_images)
    #y= y[:]

    X_train = X_train[:, :]
    y_train = y_train[:]
    Xunlabelled = Xunlabelled[:10000,:]

    #import ipdb; ipdb.set_trace()

    X_both = np.vstack((X_train, Xunlabelled))

    y_both = np.append(y_train, -np.ones((Xunlabelled.shape[0],)))


    label_prop_model = LabelSpreading(max_iter=100)
    #random_unlabeled_points = np.where(np.random.random_integers(0, 1, size=len(y_train)))
    #labels = np.copy(y_train)
    #labels[random_unlabeled_points] = -1
    label_prop_model.fit(np.copy(X_both), np.copy(y_both))
    y_pred = label_prop_model.predict(np.copy(X_both))
    print(y_pred)
    #import ipdb; ipdb.set_trace()
    #show_metrics(y_test, y_pred)


    #clf = label_propagation.LabelSpreading(max_iter=100, gamma=0.25).fit(X_both, y_both)

    #clf = LabelPropagation(max_iter=100, gamma=0.25).fit(X_train, y_train)

    #y_pred = clf.predict(X_test)

    #import ipdb; ipdb.set_trace()

    #show_metrics(y_test, y_pred)

def randomized_PCA(X_train, n_components=150, whiten=True, **params):
    #return decomposition.FactorAnalysis(n_components=n_components, max_iter=200).fit(X_train)
    return ProbabilisticPCA(n_components=n_components, whiten=whiten, **params).fit(X_train)

def ICA(X_train, n_components=150, max_iter=500, **params):
    return FastICA(n_components=n_components, max_iter=max_iter, algorithm='deflation', **params).fit(X_train)

def LDAclf(X_train, X_test, y_train, y_test, val_images=None):
    pca = randomized_PCA(X_train) 

    ae = Autoencoder(max_iter=200,sparsity_param=0.1,
                                    beta=3,n_hidden = 190,alpha=3e-3,
                                    verbose=False, random_state=1).fit(X_train)

    X_train, X_test = ae.transform(X_train), ae.transform(X_test)

    lda = LDA()
    y_pred = lda.fit(X_train, y_train).predict(X_test)
    show_metrics(y_test, y_pred)

def pca_svm(X_train, X_test, y_train, y_test):
    # X_train = divide(X_train)
    # X_test = divide(X_test)

    pca = randomized_PCA(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)


    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print clf
    y_pred = clf.predict(X_test_pca)
    show_metrics(y_test, y_pred)

    # rbf works the best
    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5], }
    # clf = GridSearchCV(SVC(kernel='linear', class_weight='auto'), param_grid)
    # clf = clf.fit(X_train_pca, y_train)
    # print clf
    # y_pred = clf.predict(X_test_pca)
    # show_metrics(y_test, y_pred)

    # param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5], 'degree':[2,3,4] }
    # clf = GridSearchCV(SVC(kernel='poly', class_weight='auto'), param_grid)
    # clf = clf.fit(X_train_pca, y_train)
    # print clf
    # y_pred = clf.predict(X_test_pca)
    # show_metrics(y_test, y_pred)

    # val_pca = pca.transform(val_images)
    # val_pred = clf.predict(val_pca)
    # make_submission_file(val_pred)

def pca_svm2(X_train, X_test, y_train, y_test, Xunlabelled, X_val=None):

    X_both = np.vstack((X_train, Xunlabelled))
    y_both = np.append(y_train, -np.ones((Xunlabelled.shape[0],)))
    pca = randomized_PCA(X_both)

    print 'done pca...'

    X_train_pca = pca.transform(X_train)

    if X_val is not None:
        X_val_pca = pca.transform(X_val)
    else:
        X_test_pca = pca.transform(X_test)

    print 'done transforming...'

    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print clf

    if X_val is not None:
        y_pred = clf.predict(X_val_pca)
        make_submission_file(y_pred)
        print 'made submission file.'
    else:
        y_pred = clf.predict(X_test_pca)
        show_metrics(y_test, y_pred)



def ica_svm(X_train, X_test, y_train, y_test, val_images=None):
    pca = ICA(X_train)

    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)

    X_train_pca = pca.transform(X_train)
    clf = clf.fit(X_train_pca, y_train)
    print clf

    X_test_pca = pca.transform(X_test)
    y_pred = clf.predict(X_test_pca)

    show_metrics(y_test, y_pred)

    val_pca = pca.transform(val_images)
    val_pred = clf.predict(val_pca)

    make_submission_file(val_pred)

def adaboost(X_train, X_test, y_train, y_test, val_images=None):
    pca = randomized_PCA(X_train, n_components=150, whiten=True)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200).fit(X_train, y_train)
    y_pred = bdt.predict(X_test)
    show_metrics(y_test, y_pred)

def autoencode_adaboost(X_train, X_test, y_train, y_test, val_images=None):
    divide = np.vectorize(lambda x: x/255.0)
    X_train = divide(X_train)
    X_test = divide(X_test)

    ae = Autoencoder(max_iter=200,sparsity_param=0.1,
                                    beta=3,n_hidden = 190,alpha=3e-3,
                                    verbose=False, random_state=1).fit(X_train)
    X_train = ae.transform(X_train)
    X_test = ae.transform(X_test)
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=200).fit(X_train, y_train)
    y_pred = bdt.predict(X_test)
    show_metrics(y_test, y_pred)

def sparse_pca(X_train, X_test, y_train, y_test, val_images=None):
    spca = decomposition.MiniBatchSparsePCA(n_components=7, alpha=0.8, ridge_alpha=0.01, n_iter=1000, n_jobs=20, verbose=True).fit(X_train, y_train)
    X_train_transform = spca.transform(X_train)
    X_test_transform = spca.transform(X_test)
    
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_transform, y_train)
    print clf

    y_pred = clf.predict(X_test_transform)    

    show_metrics(y_test, y_pred)

    knn(X_train_transform, y_train, X_test_transform, y_test)

def svm_fn(X_train, X_test, y_train, y_test, val_images=None):
    param_grid = {'C': [1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
    print clf
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    show_metrics(y_test, y_pred)

def knn(X_train, y_train, X_test, y_test):
    param_grid = {'n_neighbors': [1,3,5,7,9]}
    nbrs = GridSearchCV(KNeighborsClassifier(), param_grid)
    nbrs.fit(X_train, y_train)
    print nbrs
    y_pred = nbrs.predict(X_test)
    show_metrics(y_test, y_pred)

def preprocess(X):
    return X.astype('float64')

def pca_mlp(X_train, X_test, y_train, y_test, val_images=None):
    pca = randomized_PCA(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = {'l2decay': [0.01, 0.1, 0.5, 0.8, 1.0, 1.5] }
    clf = MLPClassifier(batch_size=129, loss='cross_entropy', output_layer='softmax', l2decay=0.1)

    clf.fit(X_train_pca, y_train, max_epochs=100)
    y_pred = clf.predict(X_test_pca)
    show_metrics(y_test, y_pred)

def main():
    np.set_printoptions(precision=None, threshold='nan', edgeitems=None, linewidth='nan', suppress=None, nanstr=None, infstr=None, formatter=None)

    X, y = load_train()
    Xval = load_valid()

    X = preprocess(X)
    Xval = preprocess(Xval)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    Xunlabelled = load_unlabelled()

    print 'data loaded...'

    #generate_arff(tr_images, tr_labels)
    #pca = randomized_PCA(tr_images)
    #X_pca = pca.transform(tr_images)
    #generate_arff(X_pca, tr_labels, 'data_pca')

    #NN(X_train, X_test, y_train, y_test, val_images)

    #label_spreading(X_train, y_train, Xunlabelled, X_test, y_test)

    pca_svm2(X_train, X_test, y_train, y_test, Xunlabelled)
    pca_svm2(X, None, y, None, Xunlabelled, Xval)


if __name__ == '__main__':
    main()