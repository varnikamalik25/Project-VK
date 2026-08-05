import numpy as np
# np.random.seed(0)
import math

class Layer:

    def __init__(self,inputs,neurons):
        self.weights = 0.10*np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))


    def softmax(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.outputs = exp_values/np.sum(exp_values,axis=1,keepdims=True)

    def ReLU(self,inputs):
        self.outputs = np.maximum(0,inputs)




    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases





# layer1 = Layer(3,3)
# layer2 = Layer(3,3)
# layer1.softmax(X)
# target_output = [[0,0,0],[0,1,0],[0,0,0]]
# softmax_outputs = layer1.outputs
# print(softmax_outputs)
# loss = 0
# for x,y in zip(target_output,softmax_outputs):
#     for i,j in zip(x,y):

#         loss += math.log(j)*i
# print(loss*-1)
# layer2.forward(layer1.outputs)
# print(layer2.outputs)

