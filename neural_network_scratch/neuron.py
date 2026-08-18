# np.random.seed(0)
import math

import decode_inputs
import numpy as np


class Layer:
    def __init__(self, inputs, neurons):
        self.weights = 0.10 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))

    def softmax(self, inputs):

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # print(exp_values)
        self.outputs = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def ReLU(self, inputs):
        self.outputs = np.maximum(0, inputs)

    def sigmoid(self, inputs):
        output = 1 / (1 + np.exp(-inputs))
        return output

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

    def loss(self, target_output, softmax_output):
        loss = 0
        for i, j in zip(target_output,softmax_output[0]):

            loss += math.log(j)*i
        return -loss
    def backprop(self, dvalues, inputs):
        self.dweights = np.dot(inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        return np.dot(dvalues, self.weights.T)

    def update_weights(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases







learning_rate = 0.01

target_outputs = [0,1]
layer1 = Layer(784, 128)
layer2 = Layer(128, 2)
factor = 1000
for i in range(100000):

    if i%factor ==0:
        target_outputs = list(map(lambda x: not x, target_outputs))
        img_path = f"../train/0/0{str(0)*(3-len(str(i//factor)))}{str(i//factor)}.png"
        # print(img_path)
        X = decode_inputs.decode_image(img_path)

        X = X.astype(np.float32)
        X /= 255.0  # Normalize

        X = X.reshape(1, 784)
    layer1.forward(X)
    hidden = layer1.sigmoid(layer1.outputs)

    layer2.forward(hidden)
    layer2.softmax(layer2.outputs)

    prediction = layer2.outputs
    # Output gradient
    d_output = prediction - target_outputs

    # Backprop layer 2
    d_hidden = layer2.backprop(d_output, hidden)

    # Backprop through sigmoid
    d_hidden *= hidden * (1 - hidden)

    # Backprop layer 1
    layer1.backprop(d_hidden, X)

    # Update weights
    layer2.update_weights(learning_rate)
    layer1.update_weights(learning_rate)

    loss = layer2.loss(target_outputs,layer2.outputs)


    print(loss)
    print(prediction)


