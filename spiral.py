import numpy
import nnfs

from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

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

layer_1 = Layer_Dense(2, 5)
activation_1 = Activation_ReLU()

layer_1.forward(X)

print(layer_1.output)
activation_1.forward(layer_1.output)
print(activation_1.output)
