from sklearn import svm
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.decomposition import ProbabilisticPCA
from sklearn.decomposition import FastICA
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from time import time
from utils import *
import numpy as np

rng = np.random.RandomState(0)
divide = np.vectorize(lambda x: x/255.0)

# Normalize data and convert to float.
def preprocess_normalize(X):
    for item in X:
        avg_pixel = np.average(item).astype('float64') + 1
        for pixel in item:
            pixel = pixel.astype('float64')/avg_pixel
    return X
    
# Convert data to float.
def preprocess(X):
    return X.astype('float64')
  
def ICA(X_train, n_components, max_iter=500, **params):
    return FastICA(n_components=n_components, max_iter=max_iter, algorithm='deflation', **params).fit(X_train)  
    
def probabilistic_PCA(X_train, n_components, whiten=True, **params):
    return ProbabilisticPCA(n_components=n_components, whiten=whiten, **params).fit(X_train)
    
# Set up training data for ICA or PCA.
def get_training_data(X_train, use_unlabelled, fraction):
    if (use_unlabelled == True):
        # Stack unlabelled data with training data.
        X_unlabelled = preprocess(load_unlabelled())
        X_unlabelled = X_unlabelled[:(len(X_unlabelled)/fraction)]
        return np.vstack((X_unlabelled, X_train))
    else:
        return X_train
    
    
# Run ICA on data.
def preprocess_ICA(X_train, X_test, use_unlabelled, n_components):
    print "*** ICA, Unlabelled =", use_unlabelled, " components =", n_components
    
    training_data = get_training_data(X_train, use_unlabelled)
    ica = ICA(training_data, n_components)
    X_train = ica.transform(X_train)
    X_test = ica.transform(X_test)
    return X_train, X_test
    
    
# Run PCA on data.
def preprocess_PCA(X_train, X_test, use_unlabelled, n_components, fraction=2):
    print "*** PCA, Unlabelled =", use_unlabelled, " components =", n_components
    
    training_data = get_training_data(X_train, use_unlabelled, fraction)
    pca = probabilistic_PCA(training_data, n_components)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test


# Run PCA_SVM, PCA_MOG, Random Forest and take majority class.
def my_ensemble(X_train, X_test, y_train, y_test):
    y_pred = []
    votes = []
    
    X_train_pca, X_test_pca = preprocess_PCA(X_train, X_test, True, 100)
    votes.append(svm(X_train_pca, X_test_pca, y_train))
    votes.append(mog(X_train_pca, X_test_pca, y_train, 10))
    votes.append(ensemble(X_train, X_test, y_train))
    
    print "*** SVM ***"
    show_metrics(votes[0], y_test)
    print "*** mog ***"
    show_metrics(votes[1], y_test)
    print "*** ensemble ***"
    show_metrics(votes[2], y_test)
    print "*** my_ensemble ***"
    
    for i in range(len(X_test)):
        if votes[1][i] == votes[2][i]:
            y_pred.append(votes[1][i])
        else:
            y_pred.append(votes[0][i])
    
    return y_pred

# Run PCA then SVM.
def pca_svm(X_train, X_test, y_train, use_unlabelled=True, pca_components=150, fraction=2):
    X_train, X_test = preprocess_PCA(X_train, X_test, use_unlabelled, pca_components, fraction)
    return svm(X_train, X_test, y_train)


# Run PCA then Ensemble.
def pca_ensemble(X_train, X_test, y_train, n_estimators=50, use_unlabelled=False, pca_components=25):
    X_train, X_test = preprocess_PCA(X_train, X_test, use_unlabelled, pca_components)
    return ensemble(X_train, X_test, y_train, n_estimators)
    
    
# Run ICA then MOG.
def ica_mog(X_train, X_test, y_train, n_components = 25, ica_components = 150, use_unlabelled = False):
    X_train, X_test = preprocess_ICA(X_train, X_test, use_unlabelled, ica_components)
    return mog(X_train, X_test, y_train, n_components)
    
    
# Run PCA then MOG.
def pca_mog(X_train, X_test, y_train, n_components = 10, pca_components = 100, use_unlabelled = True):
    X_train, X_test = preprocess_PCA(X_train, X_test, use_unlabelled, pca_components)
    return mog(X_train, X_test, y_train, n_components)
    
    
# Run SVM.
def svm(X_train, X_test, y_train, kernel='rbf'):
    print "*** SVM, kernel = ", kernel
    param_grid = {'C': [1e1, 1e2, 1e3, 1e4, 1e5],
                  'gamma': [0.0001, 0.001, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel=kernel, class_weight='auto'), param_grid)
    clf = clf.fit(X_train, y_train)
    return clf.predict(X_test)
    
    
# Run Ensemble method.
def ensemble(X_train, X_test, y_train, n_estimators=100):
    print "*** Random Forest, estimators = ", n_estimators
    
    # Random forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=None, min_samples_split=1, random_state=0)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
    
    

# Mixture of Gaussians.
def mog(X_train, X_test, y_train, n_components = 25, cov_type = 'diag'):
    print "*** MoG, components = ", n_components
    
    # Train MoG for each class and compute score.
    score_train = []
    score_test = []
    score_val = []
    min_score = 0
    classifiers = list(GMM(n_components=n_components, covariance_type=cov_type, n_iter=10) for i in range(1,8))
    
    for i in range(len(classifiers)):
        classifiers[i].fit(X_train[y_train == i + 1])
        
#        score_train.append(classifiers[i].score(X_train))
        score_test.append(classifiers[i].score(X_test))
        min_score = min(min(score_test[i]), min_score)
        
#    # Compute training error.
#    y_train_pred = []
#    for j in range(len(X_train)):
#        max_score = min_score
#        chosen_class = 0
#        for i in range(len(classifiers)):
#            if (score_train[i][j] > max_score):
#                max_score = score_train[i][j]
#                chosen_class = i + 1
#        y_train_pred.append(chosen_class)
#    
#    print '*** train pred prob ***'
#    show_metrics(y_train, y_train_pred)
    
    
    
    # Make prediction for X_test.
    y_pred = []
    for j in range(len(X_test)):
        max_score = min_score
        chosen_class = 0
        for i in range(len(classifiers)):
            if (score_test[i][j] > max_score):
                max_score = score_test[i][j]
                chosen_class = i + 1
        y_pred.append(chosen_class)
        
    return y_pred
    
    

# Baseline learner using knn.
def knn(X_train, X_test, y_train):
    nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='auto').fit(X_train, y_train)
    return nbrs.predict(X_test)



# Main
def main():
    # Load images and preprocess them.
    tr_images, tr_labels = load_train()
    val_images = load_valid()
    test_images = load_test()
    tr_images = preprocess(tr_images)
    val_images = preprocess(val_images)
    test_images = preprocess(test_images)
    
    cross_validate = False
    
    if (cross_validate == True):
        # Obtain a cross-validation set.
        print "*** Cross validation"
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr_images, tr_labels, test_size=0.25)
    else:
        X_train = tr_images
        y_train = tr_labels
        X_test = np.vstack((val_images, test_images))
        
    y_pred = pca_svm(X_train, X_test, y_train, fraction=4)
    
    if (cross_validate == True):
        show_metrics(y_test, y_pred)
    else:
        print "*** Printing output file ***"
        make_submission_file(y_pred, "submission_pca_svm_test.csv")
    
    
            
if __name__ == '__main__':
    main()
