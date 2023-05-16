import matplotlib.pyplot as plt
import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _deriv_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        return layer_2

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2*(prediction-target)
        dprediction_dlayer1 = self._deriv_sigmoid(layer_1)

        dlayer1_dbias = 1 #bc bias is just x as a function
        dlayer1_dweights = input_vector

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_Parameters(self, derror_dbias, derror_dweights):
        self.bias -= (derror_dbias * self.learning_rate)
        self.weights -= (derror_dweights * self.learning_rate)

    #train data for certain number of iterations, input_vectors and targets are arrays of vectors/numbers
    def train(self, input_vectors, targets, iterations):
        errors = []
        for currIteration in range(iterations):
            #pick random data instance
            random = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random]
            target = targets[random]

            #update parameters based on random data point
            derror_dbias, derror_dweights = self._compute_gradients(input_vector, target)
            self._update_Parameters(derror_dbias, derror_dweights)

            #error stuff
            if(currIteration % 100 == 0):
                cumulative_error = 0 #get total error by summing error for each data point

                for dataIndex in range(len(input_vectors)):
                    prediction = self._predict(input_vectors[dataIndex])
                    target = targets[dataIndex]

                    error = np.square(prediction - target)

                    cumulative_error += error

                errors.append(cumulative_error)
        return errors

#Now, test it and plot it
input_vectors = np.array([
    [3, 1.5],
    [2, 1],
    [4, 1.5],
    [3, 4],
    [3.5, 0.5],
    [2, 0.5],
    [5.5, 1],
    [1, 1],
])
targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1

neural_network = NeuralNetwork(learning_rate)

training_error = neural_network.train(input_vectors, targets, 10000)

plt.plot(training_error)
plt.xlabel("Iteration")
plt.ylabel("Cumulative error for all vectors")
plt.savefig("cumulative_error.png")


