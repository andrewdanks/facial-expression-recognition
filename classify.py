from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.semi_supervised import label_propagation
from sklearn.decomposition import RandomizedPCA, ProbabilisticPCA
from sklearn.decomposition import FastICA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from autoencoder import Autoencoder
from sklearn.linear_model import SGDClassifier
from sklearn.semi_supervised import label_propagation
from sklearn.lda import LDA
from sklearn.qda import QDA
from time import time
from utils import *
from nn import TrainNN
import numpy as np

rng = np.random.RandomState(0)
divide = np.vectorize(lambda x: x/255.0)

def label_spreading2(X, y):
    # label_spread = label_propagation.LabelSpreading(kernel='knn', alpha=1.0)
    # label_spread.fit(X, labels)

    # step size in the mesh
    h = .02

    y_30 = np.copy(y)
    y_30[rng.rand(len(y)) < 0.3] = -1
    y_50 = np.copy(y)
    y_50[rng.rand(len(y)) < 0.5] = -1

    ls30 = (label_propagation.LabelSpreading().fit(X, y_30), y_30)
    ls50 = (label_propagation.LabelSpreading().fit(X, y_50), y_50)
    ls100 = (label_propagation.LabelSpreading().fit(X, y), y)
    rbf_svc = (svm.SVC(kernel='rbf').fit(X, y), y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    for i, (clf, y_train) in enumerate((ls30, ls50, ls100, rbf_svc)):

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        import ipdb; ipdb.set_trace()

def label_spreading(X, y, val_images):
    pca = randomized_PCA(X)
    #X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr_images, tr_labels, test_size=0.3)
    #X = pca.transform(X)
    #val_images = pca.transform(val_images)
    #y= y[:]



    num_data, num_dim = X.shape

    X_both = np.empty((num_data + val_images.shape[0], num_dim))
    for i in range(num_data + val_images.shape[0]):
        if i >= num_data:
            X_both[i] = val_images[i - num_data]
        else:
            X_both[i] = X[i]

    y_both = np.append(y, -np.ones((val_images.shape[0],)))
    y_both[:50] = -1

    show_metrics(y, output_labels)

    clf = label_propagation.LabelSpreading(max_iter=100, gamma=0.25).fit(X_both, y_both)

    output_labels = clf.transduction_

    #import ipdb; ipdb.set_trace()

    y_pred = clf.predict(X)
    show_metrics(y, y_pred)

def randomized_PCA(X_train, n_components=150, whiten=True, **params):
    #return decomposition.FactorAnalysis(n_components=n_components, max_iter=200).fit(X_train)
    return ProbabilisticPCA(n_components=n_components, whiten=whiten, **params).fit(X_train)

def ICA(X_train, n_components=100, max_iter=500, **params):
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

def pca_svm(X_train, X_test, y_train, y_test, val_images=None):
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

    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5], }
    clf = GridSearchCV(SVC(kernel='linear', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print clf
    y_pred = clf.predict(X_test_pca)
    show_metrics(y_test, y_pred)

    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5], 'degree':[2,3,4] }
    clf = GridSearchCV(SVC(kernel='poly', class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print clf
    y_pred = clf.predict(X_test_pca)
    show_metrics(y_test, y_pred)

    #import ipdb; ipdb.set_trace()

    # val_pca = pca.transform(val_images)
    # val_pred = clf.predict(val_pca)
    # make_submission_file(val_pred)

def pca_svm2(X_train, y_train, val_images=None):
    pca = randomized_PCA(X_train)

    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)

    X_train_pca = pca.transform(X_train)
    clf = clf.fit(X_train_pca, y_train)
    print clf

    val_pca = pca.transform(val_images)
    val_pred = clf.predict(val_pca)

    make_submission_file(val_pred)

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

def gmm(X_train, y_train, val_images=None):
    # Try GMMs using different types of covariances.
    classifier = GMM(n_components=150)
    # Train the other parameters using the EM algorithm.
    classifier.fit(X_train)
    y_train_pred = classifier.predict(X_train)
    show_metrics(y_train_pred, y_train)

def pca_gmm(X_train, y_train, val_images=None):
    #pca = randomized_PCA(X_train, n_components=150, whiten=True)
    gmm(X_train, y_train, val_images=None)

def preprocess(X):
    #import ipdb; ipdb.set_trace()
    return X.astype('float64')

    #return X / 255.0

    # def crop(Xi):
    #     square = Xi.reshape(32,32)
    #     return [square[:-15,:].reshape(17*32)] + [square[:-15,:].reshape(17*32)] + [square[15:,:].reshape(17*32)]

    # new_X = np.zeros((len(X), 3*17*32))
    # for i in range(len(X)):
    #     new_X[i] = crop(X[i])

    # return new_X

    # def new_val(val, step, max_size=255):
    #     i, j = 0, step
    #     bin_num = 1
    #     while i <= max_size:
    #         if val >= i and val <= j:
    #             return bin_num
    #         else:
    #             i += step
    #             j += step
    #             bin_num += 1

    # new_val_vec = np.vectorize(lambda val: new_val(val, 2))
    # X = new_val_vec(X)
    # return X 

def NN(X_train, X_test, y_train, y_test, val_images=None):
    divide = np.vectorize(lambda x: x/255.0)
    X_train = divide(X_train)
    X_test = divide(X_test)

    pca = randomized_PCA(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    num_hiddens = 10
    eps = 0.005
    momentum = 0.0
    num_epochs = 200
    TrainNN(num_hiddens, eps, momentum, num_epochs, X_train.T, X_test.T, y_train, y_test)

def generate_arff(X, y):
    pass

def main():
    np.set_printoptions(precision=None, threshold='nan', edgeitems=None, linewidth='nan', suppress=None, nanstr=None, infstr=None, formatter=None)

    tr_images, tr_labels = load_train()

    val_images = load_valid()

    tr_images = preprocess(tr_images)
    val_images = preprocess(val_images)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr_images, tr_labels, test_size=0.25)

    pca_svm(X_train, X_test, y_train, y_test, val_images)


if __name__ == '__main__':
    main()