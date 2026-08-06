import numpy as np
# np.random.seed(0)
import math
import decode_inputs


class Layer:

    def __init__(self,inputs,neurons):
        self.weights = 0.10*np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))


    def softmax(self,inputs):
        print(inputs)
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        # print(exp_values)
        self.outputs = exp_values/np.sum(exp_values,axis=1,keepdims=True)

    def ReLU(self,inputs):
        self.outputs = np.maximum(0,inputs)




    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases





img_path = "../train/0/0000.png"
X = decode_inputs.decode_image(img_path)


layer1 = Layer(28,28)
layer2 = Layer(28,28)
layer1.softmax(X)

softmax_outputs = layer1.outputs
# print(softmax_outputs)
# loss = 0
# for x,y in zip(target_output,softmax_outputs):
#     for i,j in zip(x,y):

#         loss += math.log(j)*i
# print(loss*-1)
print()
layer2.forward(layer1.outputs)
# print(layer2.outputs)

