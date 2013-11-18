"""Sparse Autoencoder
"""

# Author: Issam Laradji <issam.laradji@gmail.com>

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils import atleast2d_or_csr, check_random_state
from sklearn.base import BaseEstimator, TransformerMixin

def _binary_KL_divergence(p, p_hat):
    return (p * np.log(p / p_hat)) + ((1 - p) * np.log((1 - p) / (1 - p_hat)))

def _logistic(X):
    return 1. / (1. + np.exp(np.clip(-X, -30, 30)))
    
def _d_logistic(X):
    return X * (1 - X)
    
class Autoencoder(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_hidden=25,
        learning_rate=0.3, alpha=3e-3, beta=3, sparsity_param=0.1,
        max_iter=20, verbose=False, random_state=None):
        self.n_hidden = n_hidden
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.beta = beta
        self.sparsity_param = sparsity_param
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def _init_fit(self, n_features):
        """
        Weights' initilization
        """
        rng = check_random_state(self.random_state)
        self.coef_hidden_ = rng.uniform(-1, 1, (n_features, self.n_hidden))
        self.coef_output_ = rng.uniform(-1, 1, (self.n_hidden, n_features))
        self.intercept_hidden_ = rng.uniform(-1, 1, self.n_hidden)
        self.intercept_output_ = rng.uniform(-1, 1, n_features)

    def _unpack(self, theta, n_features):
        N = self.n_hidden * n_features
        self.coef_hidden_ = np.reshape(theta[:N],
                                      (n_features, self.n_hidden))
        self.coef_output_ = np.reshape(theta[N:2 * N],
                                      (self.n_hidden, n_features))
        self.intercept_hidden_ = theta[2 * N:2 * N + self.n_hidden]
        self.intercept_output_ = theta[2 * N + self.n_hidden:]

    def _pack(self, W1, W2, b1, b2):
        return np.hstack((W1.ravel(), W2.ravel(),
                          b1.ravel(), b2.ravel()))

    def transform(self, X):
        return _logistic(np.dot(X, self.coef_hidden_) + self.intercept_hidden_)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None):
        n_samples, n_features = X.shape
        self._init_fit(n_features)
        self._backprop_lbfgs(
                X, n_features, n_samples)
        return self

    def backprop(self, X, n_features, n_samples):
        # Forward pass
        a_hidden = _logistic(np.dot(X, self.coef_hidden_)
                                      + self.intercept_hidden_)
        a_output = _logistic(np.dot(a_hidden, self.coef_output_)
                                      + self.intercept_output_)
        # Compute average activation of hidden neurons
        p = self.sparsity_param
        p_hat = np.mean(a_hidden, 0)
        p_delta  = self.beta * ((1 - p) / (1 - p_hat) - p / p_hat)
        # Compute cost 
        diff = X - a_output
        cost = np.sum(diff ** 2) / (2 * n_samples)
        # Add regularization term to cost 
        cost += (0.5 * self.alpha) * (
            np.sum(self.coef_hidden_ ** 2) + np.sum(
                self.coef_output_ ** 2))
        # Add sparsity term to the cost
        cost += self.beta * np.sum(
            _binary_KL_divergence(
                p, p_hat))
        # Compute the error terms (delta)
        delta_output = -diff * _d_logistic(a_output)
        delta_hidden = (
            (np.dot(delta_output, self.coef_output_.T) +
             p_delta)) * _d_logistic(a_hidden)
        #Get gradients
        W1grad = np.dot(X.T, delta_hidden) / n_samples 
        W2grad = np.dot(a_hidden.T, delta_output) / n_samples
        b1grad = np.mean(delta_hidden, 0) 
        b2grad = np.mean(delta_output, 0) 
        # Add regularization term to weight gradients 
        W1grad += self.alpha * self.coef_hidden_
        W2grad += self.alpha * self.coef_output_
        return cost, W1grad, W2grad, b1grad, b2grad

    def _backprop_lbfgs(self, X, n_features, n_samples):
        #Pack the initial weights 
        #into a vector
        initial_theta = self._pack(
            self.coef_hidden_,
            self.coef_output_,
            self.intercept_hidden_,
            self.intercept_output_)
        #Optimize the weights using l-bfgs
        optTheta, _, _ = fmin_l_bfgs_b(
            func=self._cost_grad,
            x0=initial_theta,
            maxfun=self.max_iter,
            disp=self.verbose,
            args=(X,
                n_features,
                n_samples))
        #Unpack the weights into their
        #relevant variables
        self._unpack(optTheta, n_features)

    def _cost_grad(self, theta, X, n_features,
                   n_samples):
        self._unpack(theta, n_features)
        cost, W1grad, W2grad, b1grad, b2grad = self.backprop(
            X, n_features, n_samples)
        return cost, self._pack(W1grad, W2grad, b1grad, b2grad)

if __name__ == '__main__':
    from sklearn.linear_model import SGDClassifier
    from sklearn.datasets import fetch_mldata
    import random
    #Download dataset
    mnist = fetch_mldata('MNIST original')
    #Get 1000 samples for 'X', and 'y'
    X, y = mnist.data, mnist.target
    random.seed(1)
    indices = np.array(random.sample(range(70000), 1000))
    X, y = X[indices].astype('float64'), y[indices]
    # Scale values in the range [0, 1]
    X = X / 255
    # Set autoencoder parameters
    ae = Autoencoder(max_iter=200,sparsity_param=0.1,
                                    beta=3,n_hidden = 190,alpha=3e-3,
                                    verbose=True, random_state=1)
    # Train and extract features
    extracted_features = ae.fit_transform(X)
    # Test
    clf = SGDClassifier(random_state=3)
    clf.fit(X, y)
    print 'SGD on raw pixels score: ', \
              clf.score(X, y)
    clf.fit(extracted_features, y)
    print 'SGD on extracted features score: ', \
              clf.score(extracted_features, y)
