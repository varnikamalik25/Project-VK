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






img_path = "../train/0/0000.png"
X = decode_inputs.decode_image(img_path)

X = X.astype(np.float32)
X /= 255.0  # Normalize

X = X.reshape(1, 784)

target_outputs = [0,1]
layer1 = Layer(784, 128)
layer2 = Layer(128, 2)

layer1.forward(X)

layer2.softmax(layer2.sigmoid(layer1.outputs))

layer1.forward(X)

hidden = layer1.sigmoid(layer1.outputs)

layer2.forward(hidden)

layer2.softmax(layer2.outputs)

prediction = layer2.outputs

loss = layer2.loss(target_outputs,layer2.outputs)

print(loss)
print(prediction)


