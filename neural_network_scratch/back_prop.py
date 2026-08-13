import numpy as np

dvalues = np.array([[1.,1.,1.],
                    [2.,2.,2.],
                    [3.,3.,3.]])

inputs = np.array([[1,2,3,2.5],
                   [2.,5.,-1.,2],
                   [-1.5,2.,3.,0.8]])

weights = np.array([[0.2,0.8,-0.5,1],
                    [0.5,-0.91,0.26,-0.5],
                    [-0.26,-0.27,0.17,0.87]])

biases = np.array([[2,3,0.5]])

l_out = np.dot(inputs,weights.T) + biases
relu_out = np.maximum(0,l_out)
drelu = relu_out.copy()
drelu[l_out <= 0] = 0
dinputs = np.dot(drelu,weights)
dweights = np.dot(inputs.T,drelu)
dbiases = np.sum(drelu,axis = 0, keepdims = True)

# weights += -0.001*dweights
# biases += -0.001*dbiases
print(dweights)
print()
print(dbiases)
