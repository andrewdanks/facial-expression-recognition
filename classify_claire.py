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
from sklearn.linear_model import SGDClassifier
from sklearn.semi_supervised import label_propagation
from sklearn.lda import LDA
from sklearn.qda import QDA
from time import time
from utils import *
import numpy as np

rng = np.random.RandomState(0)
divide = np.vectorize(lambda x: x/255.0)

# Convert data to float.
def preprocess(X):
    return X.astype('float64')

def randomized_PCA(X_train, n_components=150, whiten=True, **params):
    return ProbabilisticPCA(n_components=n_components, whiten=whiten, **params).fit(X_train)
    
    
    
# Run PCA on data.
def preprocess_PCA(X_train, X_test, use_unlabelled = True):
    if (use_unlabelled == True):
        # Run PCA using unlabelled data.
        X_unlabelled = load_unlabelled()
        unlabelled_size = len(X_unlabelled)/2
        print "PCA - Using unlabelled data: ", unlabelled_size
        X_unlabelled = X_unlabelled[:unlabelled_size]
        training_data = np.vstack((X_unlabelled, X_train))
    else:
        print "PCA"
        training_data = X_train
        
    pca = randomized_PCA(training_data)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test



# Run PCA then MOG.
def pca_mog(X_train, X_test, y_train, y_test, val_images, use_unlabelled = True, n_components = 25):
    if (X_test == None):
        X_train, val_images = preprocess_PCA(X_train, val_images, use_unlabelled)
    else:
        X_train, X_test = preprocess_PCA(X_train, X_test, use_unlabelled)
    mog(X_train, X_test, y_train, y_test, val_images, False, n_components)
    
    

# Mixture of Gaussians.
def mog(X_train, X_test, y_train, y_test, val_images, x, n_components = 25, cov_type = 'diag'):
    print "Number of Components = ", n_components
    print "Covariance Type = ", cov_type
    
    # Train MoG for each class and compute score.
    score_train = []
    score_test = []
    score_val = []
    min_score = 0
    classifiers = list(GMM(n_components=n_components, covariance_type=cov_type, n_iter=10) for i in range(1,8))
    
    for i in range(len(classifiers)):
        classifiers[i].fit(X_train[y_train == i + 1])
        
        score_train.append(classifiers[i].score(X_train))
        if (X_test == None):
            score_val.append(classifiers[i].score(val_images))
            min_score = min(min(score_val[i]), min_score)
        else:
            score_test.append(classifiers[i].score(X_test))
            min_score = min(min(score_test[i]), min_score)
        
    # Compute training error.
    y_train_pred = []
    for j in range(len(X_train)):
        max_score = min_score
        chosen_class = 0
        for i in range(len(classifiers)):
            if (score_train[i][j] > max_score):
                max_score = score_train[i][j]
                chosen_class = i + 1
        y_train_pred.append(chosen_class)
    
#    print '*** train pred prob ***'
#    show_metrics(y_train, y_train_pred)
    
    
    if (X_test == None):
        # Produce prediction file using validation images.
        val_pred = []
        for j in range(len(val_images)):
            max_score = min_score
            chosen_class = 0
            for i in range(len(classifiers)):
                if (score_val[i][j] > max_score):
                    max_score = score_val[i][j]
                    chosen_class = i + 1
            val_pred.append(chosen_class)
        print "*** Producing output file ***"
        make_submission_file(val_pred, "submission_pca_mog.csv")
    else:
        # Compute validation error.
        y_pred = []
        for j in range(len(X_test)):
            max_score = min_score
            chosen_class = 0
            for i in range(len(classifiers)):
                if (score_test[i][j] > max_score):
                    max_score = score_test[i][j]
                    chosen_class = i + 1
            y_pred.append(chosen_class)
            
        print '*** pred prob ***'
        show_metrics(y_test, y_pred)
    
    

# Baseline learner using knn.
def knn(X_train, X_test, y_train, y_test, val_images):
    nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='auto').fit(X_train, y_train)
    print nbrs
    if (X_test == None):
        val_pred = nbrs.predict(val_images)
        make_submission_file(val_pred, "submission_knn.csv")
    else:
        y_pred = nbrs.predict(X_test)
        show_metrics(y_test, y_pred)



# Main
def main():
    # Load images and preprocess them.
    tr_images, tr_labels = load_train()
    val_images = load_valid()
    tr_images = preprocess(tr_images)
    val_images = preprocess(val_images)
    cross_validate = False
    
    if (cross_validate == True):
        # Obtain a cross-validation set.
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr_images, tr_labels, test_size=0.25)
    else:
        X_train = tr_images
        y_train = tr_labels
        X_test = None
        y_test = None
        
    pca_mog(X_train, X_test, y_train, y_test, val_images, True, 15)

if __name__ == '__main__':
    main()
