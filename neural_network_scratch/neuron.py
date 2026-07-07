import numpy as np
np.random.seed(0)

class Layer:
    
    def __init__(self,inputs,neurons):
        self.weights = 0.10*np.random.randn(inputs,neurons)
        self.biases = np.zeros((1,neurons))

    def forward(self,inputs):
        self.outputs = np.dot(inputs,self.weights) + self.biases
        
        

X = [[1,2,3,2.5],
[2.0,5.0,-1.0,2.0],
[-1.5,2.7,3.3,-0.8]]

layer1 = Layer(4,5)
layer2 = Layer(5,2)

layer1.forward(X)
layer2.forward(layer1.outputs)
print(layer2.outputs)
