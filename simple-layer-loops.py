# Inputs, weights, and biases
inputs = [1, 2, 3, 2.5]
weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5], 
            [-0.26, -0.27, 0.17, 0.87] ]
biases = [2, 3, 0.5]

# Output calculation (with loops)
layer_outputs = []

for neuron_weights, neuron_bias in zip(weights, biases):
  neuron_output = 0

  for neuron_input, neuron_weight in zip(inputs, neuron_weights):
    neuron_output += neuron_input * neuron_weight

  neuron_output += neuron_bias
  layer_outputs.append(neuron_output)

# Print out the outputs
print(layer_outputs)