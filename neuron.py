# Basic neuron code - somewhere in our fully connected neural network
# Every neuron has a unique connection to each previous neuron

# The outputs of the previous neurons are the inputs of this one
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3

output = (inputs[0] * weights[0] 
  + inputs[1] * weights[1] 
  + inputs[2] * weights[2] 
  + bias)

print(output)