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

# Convert data to float.
def preprocess(X):
	return X.astype('float64')

def randomized_PCA(X_train, n_components=150, whiten=True, **params):
    return ProbabilisticPCA(n_components=n_components, whiten=whiten, **params).fit(X_train)
    
def preprocess_PCA(X_train, X_test):
	pca = randomized_PCA(X_train)
	X_train = pca.transform(X_train)
	X_test = pca.transform(X_test)
	return X_train, X_test

# Mixture of Gaussians.
# n_components = 50 with PCA:
#	precision = 0.60, recall = 0.57, f1-score = 0.56
def mog(X_train, X_test, y_train, y_test, val_images):
	# Train MoG for each class and compute score.
	score_train = []
	score_val = []
	min_score = 0
	n_components = 50
	print "n_components = ", n_components
	classifiers = list(GMM(n_components=n_components, covariance_type='diag', n_iter=5) for i in range(1,8))
	for i in range(len(classifiers)):
		classifiers[i].fit(X_train[y_train == i + 1])
		score_train.append(classifiers[i].score(X_train))
		score_val.append(classifiers[i].score(X_test))
		min_score = min(min(score_val[i]), min_score)
		
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
	
	print '*** train pred prob ***'
	show_metrics(y_train, y_train_pred)
	
	# Compute validation error.
	y_pred = []
	for j in range(len(X_test)):
		max_score = min_score
		chosen_class = 0
		for i in range(len(classifiers)):
			if (score_val[i][j] > max_score):
				max_score = score_val[i][j]
				chosen_class = i + 1
		y_pred.append(chosen_class)
		
	print '*** pred prob ***'
	show_metrics(y_test, y_pred)
	

# Baseline learner using knn.
# precision = 0.5, recall = 0.5, f1-score = 0.47
def knn(X_train, X_test, y_train, y_test, val_images):
	nbrs = KNeighborsClassifier(n_neighbors=5, algorithm='auto').fit(X_train, y_train)
	print nbrs
	y_pred = nbrs.predict(X_test)
	show_metrics(y_test, y_pred)
	val_pred = nbrs.predict(val_images)
	make_submission_file_2(val_pred, "submission_knn.csv")

def main():
    tr_images, tr_labels = load_train()

    val_images = load_valid()

    tr_images = preprocess(tr_images)
    val_images = preprocess(val_images)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr_images, tr_labels, test_size=0.25)
    
    X_train, X_test = preprocess_PCA(X_train, X_test)

    mog(X_train, X_test, y_train, y_test, val_images)


if __name__ == '__main__':
    main()
