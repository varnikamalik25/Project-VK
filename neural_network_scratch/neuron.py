import numpy as np

class Neuron:
    
    weights = []
    bias = 0
    
    def __init__(self,weight,bias):
        
        
        self.weights = weight
        self.bias = bias

    def validate_inputs(self,i,w):
        return  (len(i) == len(w))
    def output(self,inputs):
        return np.dot(self.weights,inputs)+self.bias
        

inputs = [1,2,3,2.5]
neuron1 = Neuron([0.2,0.8,-0.5,1.0],2)
neuron2 = Neuron([0.5,-0.91,0.26,-0.5],3)
neuron3 = Neuron([-0.26,-0.27,0.17,0.87],0.5)

print([neuron1.output(inputs),neuron2.output(inputs),neuron3.output(inputs)])