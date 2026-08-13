import numpy as np

import math
from PIL import Image
np.random.seed(0)

img=Image.open('../test/1/0000.png')
X=np.array(img.getdata()).reshape(1,784)

print("X INPUT: ")
print(X,X.shape)
print()

class Layer_Dense:
    def __init__(self,n_input,n_neuron):
        self.weight=0.05*np.random.randn(n_input,n_neuron)
        self.bias=np.zeros((1,n_neuron))

    def forward(self,input):
        self.output=np.dot(input,self.weight)+self.bias

    def softmax(self, inputs):
        exp_values=np.exp(inputs - np.max(inputs,axis=1, keepdims=True))
        self.output=exp_values/np.sum(exp_values)

    def sigmoid(self,inputs):
        self.output=1/(1+np.exp(-inputs))

def loss_calculation(real_output,target_output):
    loss=0;
    for i,j in zip(real_output[0],target_output):
        loss+=math.log(i)*j
    return loss*(-1)

target_values=[0,1]

dense1=Layer_Dense(784,128)
dense2=Layer_Dense(128,2)

dense1.forward(X)

dense2.sigmoid(dense1.output)
#     doing sigmoid activation on the output of dense1 layer to make sure the values are between 0 and 1

dense1.softmax(dense1.output)
soft=dense1.output
print(soft,soft.shape)

dense2.forward(dense1.output)

dense2.softmax(dense2.output)
soft2=dense2.output
print(soft2,soft2.shape)

loss=loss_calculation(soft2,target_values)
print(loss)
