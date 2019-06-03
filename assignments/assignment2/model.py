import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.dense_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.dense_2 = FullyConnectedLayer(hidden_layer_size, n_output)
        self.relu_1 = ReLULayer()
        self.relu_2 = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Zero grad
        params = self.params()
        for param in params:
            params[param].grad = np.zeros_like(params[param])
        
        # Forward pass
        x = self.dense_1.forward(X)
        x = self.relu_1.forward(x)
        x = self.dense_2.forward(x)
        x = self.relu_2.forward(x)

        loss, grad = softmax_with_cross_entropy(x, y)

        l = self.relu_2.backward(grad)
        l = self.dense_2.backward(l)
        l = self.relu_1.backward(l)
        l = self.dense_1.backward(l)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        l2 = 0.0
        params = self.params()
        for param in params:
            l2_loss, l2_grad = l2_regularization(params[param].value, float(self.reg))
            l2 += l2_loss
            params[param].grad = l2_grad

        return loss + l2

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        pred = np.zeros(X.shape[0], np.int)

        pred = self.dense_1.forward(X)
        pred = self.relu_1.forward(pred)
        pred = self.dense_2.forward(pred)
        pred = self.relu_2.forward(pred)

        return np.argmax(softmax(pred), axis=1)

    def params(self):
        result = {}

        result['dense_1_W'] = self.dense_1.W
        result['dense_1_B'] = self.dense_1.B

        result['dense_2_W'] = self.dense_2.W
        result['dense_2_B'] = self.dense_2.B

        return result
