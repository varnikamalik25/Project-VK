import numpy as np
# np.random.seed(0)
import math
import decode_inputs


class Layer:

    def __init__(self,inputs,neurons):
        self.weights = 0.10*np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))


    def softmax(self,inputs):

        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        # print(exp_values)
        self.outputs = exp_values/np.sum(exp_values,axis=1,keepdims=True)

    def ReLU(self,inputs):
        self.outputs = np.maximum(0,inputs)

    def sigmoid(self,inputs):
        output = 1/(1+np.exp(-inputs))
        return output




    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases





img_path = "../train/0/0000.png"
X = decode_inputs.decode_image(img_path)

X = X.astype(np.float32)
X /= 255.0          # Normalize

X = X.reshape(1,784)



layer1 = Layer(784,128)
layer2 = Layer(128,2)

layer1.forward(X)

layer2.softmax(layer2.sigmoid(layer1.outputs))

layer1.forward(X)

hidden = layer1.sigmoid(layer1.outputs)

layer2.forward(hidden)

layer2.softmax(layer2.outputs)

prediction = layer2.outputs
print(prediction)


# print(softmax_outputs)
# loss = 0
# for x,y in zip(target_output,softmax_outputs):
#     for i,j in zip(x,y):

#         loss += math.log(j)*i
# print(loss*-1)

# print(layer2.outputs)

