import numpy as np
np.random.seed(0)

class Layer:

    def __init__(self,inputs,neurons):
        self.weights = 0.10*np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))


    def take_exponent(self,inputs):
        exp_values = np.sin(inputs-np.max(inputs,axis=1,keepdims=True))
        self.outputs = exp_values/np.sum(exp_values,axis=1,keepdims=True)




    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases



X = [[1,2,3],
[2.0,5.0,-1.0],
[-1.5,2.7,3.3]]

layer1 = Layer(3,3)
layer2 = Layer(3,3)

layer1.take_exponent(X)
#print(layer1.outputs)
layer2.forward(layer1.outputs)
print(layer2.outputs)

