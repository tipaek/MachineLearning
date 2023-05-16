import numpy as np
import random

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# Computing the dot product of input_vector and weights_1
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult

print(f"Dot product from basic math: {dot_product_1}")
print(f"Dot product from function: {np.dot(input_vector, weights_1)}")
print("Dot product b/w input and weight_2: " + str(np.dot(input_vector, weights_2)))


#Now to solve basic classification problem(dot product layer and sigmoid function layer)

#Wrap vectors in NumPy arrays
input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction = make_prediction(input_vector, weights_1, bias)


print(f"\n\n\nThe prediction result is {prediction}")

prediction2 = make_prediction(np.array([2, 1.5]), weights_1, bias)
print(f"Now for the other input vector: {prediction2}")

#prediction2 returned 0.87 when we wanted 0, meaning it's wrong. Therefore, use mean squared error to quantify error
print(f"The error magnitude based on MSE for prediction 2 is: {np.square(prediction2 - 0)}")

#use gradient descent to find direction to change weights
gradient = 2*(prediction2 - 0)
print("The gradient is: " + str(gradient))

#It's positive, so decrease the weight
weights_1 -= gradient
prediction2 = make_prediction(np.array([2, 1.5]), weights_1, bias)
print("\n\n\nAfter updating, the prediction is: " + str(prediction2))
print(f"The error magnitude based on MSE for prediction 2 after updating is: {np.square(prediction2 - 0)}")

print(f"Now checking weight on vector one again... {make_prediction(input_vector, weights_1, bias)}")
print("It's wrong, so try to fix that")

#Now, we're going to implement backpropagation by using partial derivatives and chain rule following the flow chart
print("\n\n\nNow we're going to try to implement backpropagation...")
target = 0
def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

derror_dprediction = 2*(prediction - target) #derivative of error function
layer_1 = np.dot(input_vector, weights_1) #just calculating
dprediction_dlayer1 = sigmoid_deriv(layer_1) #derivative of prediction function with respect to layer 1 is just dsigmoid
dlayer1_dbias = 1

derror_dbias = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
)

dlayer1_dweights = 1 #bro i don't get this man ahhhh



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
                cummulative_error = 0 #get total error by summing error for each data point

                for dataIndex in range(len(input_vectors)):
                    prediction = self._predict(input_vectors[dataIndex])
                    target = targets[dataIndex]

                    error = np.square(prediction - target)

                    cummulative_error += error

                errors.append(cummulative_error)
        return errors

learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
print(neural_network._predict(np.array([2, 1.5])))





