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


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W * W)

    grad = reg_strength * 2 * W

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    p = preds.copy()
    target = target_index
    if len(preds.shape) == 1:
        p = p.reshape(1, -1)
    if type(target_index) == int:
        target = np.array([target_index])

    probs = softmax(p)
    loss = cross_entropy_loss(probs, target)

    y = np.eye(p.shape[1])[target]
    y = y.reshape(p.shape[0], -1)
    dprediction = probs - y

    if len(preds.shape) == 1:
        dprediction = dprediction.reshape(-1)
    else:
        dprediction /= p.shape[0]

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.index = X > 0
        return X * self.index

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return self.index.astype(float) * d_out

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.sum(d_out, axis=0, keepdims=True)

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}
