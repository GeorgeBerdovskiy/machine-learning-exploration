import numpy
import nnfs

X = [[1, 2, 3, 2.5],
	 [2, 5, -1, 2], 
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

for i in inputs:
	if i > 0:
		output.append(i)
	elif i <= 0:
		output.append(0)

	# OR output.append(max(0, i))

print(output)

class Layer_Dense:

	def __init__(self, input_count, neuron_count):
		# Multiplied by 0.10 so that (most) of the | weights | will be less than 1
		self.weights = 0.10 * numpy.random.randn(input_count, neuron_count)
		self.biases = numpy.zeros((1, neuron_count))

	def forward(self, inputs):
		# No need to transpose self.weights - already in correct orientation
		self.output = numpy.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.output = numpy.maximum(0, inputs)

layer_1 = Layer_Dense(4, 5)
layer_2 = Layer_Dense(5, 2)

layer_1.forward(X)
print(layer_1.output)

print("\n-----\n")

layer_2.forward(layer_1.output)
print(layer_2.output)