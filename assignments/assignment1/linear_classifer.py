import numpy as np


def softmax(predictions: np.array):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    '''
    if len(predictions.shape) == 1:
        probs = predictions.copy() - np.max(predictions)
        result = np.exp(probs) / np.sum(np.exp(probs))
    else:
        max_array = np.repeat(np.max(predictions, axis=1), predictions.shape[1]).reshape(predictions.shape)
        probs = predictions.copy() - max_array
        result = np.exp(probs) / np.repeat(np.sum(np.exp(probs), axis=1), predictions.shape[1]).reshape(predictions.shape)

    return result


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    probs_copy = probs.copy()
    if type(target_index) == int and len(probs.shape) == 1:
        return -np.log(probs_copy[target_index])
    else:
        length = target_index.shape[0]
        log_likelihood = -np.log(
            probs[range(length), target_index.reshape(1, -1)])

        return np.sum(log_likelihood) / length


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    p = predictions.copy()
    target = target_index
    if len(predictions.shape) == 1:
        p = p.reshape(1, -1)
    if type(target_index) == int:
        target = np.array([target_index])

    probs = softmax(p)
    loss = cross_entropy_loss(probs, target)

    y = np.eye(p.shape[1])[target]
    y = y.reshape(p.shape[0], -1)
    dprediction = probs - y

    if len(predictions.shape) == 1:
        dprediction = dprediction.reshape(-1)
    else:
        dprediction /= p.shape[0]

    return loss, dprediction


def l2_regularization(W: np.ndarray, reg_strength: float):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    loss = reg_strength * np.sum(W * W)

    grad = reg_strength * 2 * W

    return loss, grad


def linear_softmax(X: np.ndarray, W: np.ndarray, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    loss, grad_loss = softmax_with_cross_entropy(predictions, target_index)
    dW = X.T.dot(grad_loss)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier

        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            for batch in batches_indices:
                loss, gradient = linear_softmax(X[batch], self.W, y[batch])
                l2, grad_l2 = l2_regularization(self.W, reg)
                loss += l2
                gradient += grad_l2

                self.W -= learning_rate * gradient

            loss, _ = linear_softmax(X, self.W, y)
            l2, _ = l2_regularization(self.W, reg)
            loss += l2

            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        y_pred = softmax(X.dot(self.W))
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred
