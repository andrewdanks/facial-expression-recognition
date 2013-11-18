import numpy as np
import sys

def InitNN(num_inputs, num_hiddens, num_outputs):
  """Initializes NN parameters."""
  W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
  W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
  b1 = np.zeros((num_hiddens, 1))
  b2 = np.zeros((num_outputs, 1))
  return W1, W2, b1, b2

def ClassificationError(target, prediction):
    (N,) = target.shape
    num_correct = 0.0
    for i in xrange(N):
      prob = prediction[0][i]
      if (prob > 0.5 and target[i] == 1.0) or (prob <= 0.5 and target[i] == 0.0):
        num_correct += 1.0
    #import ipdb; ipdb.set_trace()
    return 1.0 - num_correct / N
    
    # num_correct = np.sum((prediction > 0.5) == True) + np.sum((prediction <= 0.5) == False)
    # error = 1 - 1.0 * num_correct / N
    # return error

def TrainNN(num_hiddens, eps, momentum, num_epochs, inputs_train, inputs_valid, target_train, target_valid):
  """Trains a single hidden layer NN.

  Inputs:
    num_hiddens: NUmber of hidden units.
    eps: Learning rate.
    momentum: Momentum.
    num_epochs: Number of epochs to run training for.

  Returns:
    W1: First layer weights.
    W2: Second layer weights.
    b1: Hidden layer bias.
    b2: Output layer bias.
    train_error: Training error at at epoch.
    valid_error: Validation error at at epoch.
  """

  W1, W2, b1, b2 = InitNN(inputs_train.shape[0], num_hiddens, target_train.shape[0])
  dW1 = np.zeros(W1.shape)
  dW2 = np.zeros(W2.shape)
  db1 = np.zeros(b1.shape)
  db2 = np.zeros(b2.shape)
  train_error = []
  valid_error = []
  train_classification_error, valid_classification_error = [], []
  num_train_cases = inputs_train.shape[1]
  for epoch in xrange(num_epochs):
    # Forward prop
    h_input = np.dot(W1.T, inputs_train) + b1  # Input to hidden layer.
    h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
    logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
    prediction = 1 / (1 + np.exp(-logit))  # Output prediction.

    # Compute cross entropy
    train_CE = -np.mean(target_train * np.log(prediction) + (1 - target_train) * np.log(1 - prediction))    

    # Compute deriv
    dEbydlogit = prediction - target_train

    # Backprop
    dEbydh_output = np.dot(W2, dEbydlogit)
    dEbydh_input = dEbydh_output * h_output * (1 - h_output)

    # Gradients for weights and biases.
    dEbydW2 = np.dot(h_output, dEbydlogit.T)
    dEbydb2 = np.sum(dEbydlogit, axis=1).reshape(-1, 1)
    dEbydW1 = np.dot(inputs_train, dEbydh_input.T)
    dEbydb1 = np.sum(dEbydh_input, axis=1).reshape(-1, 1)

    #%%%% Update the weights at the end of the epoch %%%%%%
    dW1 = momentum * dW1 - (eps / num_train_cases) * dEbydW1
    dW2 = momentum * dW2 - (eps / num_train_cases) * dEbydW2
    db1 = momentum * db1 - (eps / num_train_cases) * dEbydb1
    db2 = momentum * db2 - (eps / num_train_cases) * dEbydb2

    W1 = W1 + dW1
    W2 = W2 + dW2
    b1 = b1 + db1
    b2 = b2 + db2

    valid_CE, valid_ClsErr = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)

    train_ClsErr = ClassificationError(target_train, prediction)
    train_classification_error.append(train_ClsErr)
    valid_classification_error.append(valid_ClsErr)

    train_error.append(train_CE)
    valid_error.append(valid_CE)
    sys.stdout.write('\rStep %d Train CE %.5f Validation CE %.5f' % (epoch, train_CE, valid_CE))
    sys.stdout.flush()
    if (epoch % 100 == 0):
      sys.stdout.write('\n')

  sys.stdout.write('\n')
  final_train_error, final_train_classification_error = Evaluate(inputs_train, target_train, W1, W2, b1, b2)
  final_valid_error, final_valid_classification_error = Evaluate(inputs_valid, target_valid, W1, W2, b1, b2)

  print 'Error: Train %.5f Validation %.5f' % (final_train_error, final_valid_error)
  print 'Mean Classification Error: Train %.5f Validation %.5f' % (np.mean(train_classification_error), np.mean(valid_classification_error))
  print 'Final Classification Error: Train %.5f Validation %.5f' % (final_train_classification_error, final_valid_classification_error)
  return W1, W2, b1, b2, train_error, valid_error, train_classification_error, valid_classification_error

def Evaluate(inputs, target, W1, W2, b1, b2):
  """Evaluates the model on inputs and target."""
  h_input = np.dot(W1.T, inputs) + b1  # Input to hidden layer.
  h_output = 1 / (1 + np.exp(-h_input))  # Output of hidden layer.
  logit = np.dot(W2.T, h_output) + b2  # Input to output layer.
  prediction = 1 / (1 + np.exp(-logit))  # Output prediction.
  CE = -np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction))
  return CE, ClassificationError(target, prediction)

