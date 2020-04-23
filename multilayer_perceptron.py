import numpy as np
import matplotlib.pyplot as plt


class NeuronLayer():
	def __init__(self, inputs_length, neurons_amount):
		self.synaptic_weights = np.random.random((inputs_length, neurons_amount))
		self.bias = [-1.5 for i in range(neurons_amount)]


class NeuralNetwork():
	def __init__(self, layer_model):
		self.layers = layer_model
		self.learn_rate = 0.5
		self.errors = []

	def sigmoid_function(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivate(self, x):
		return x * (1 - x)

	def backpropagation(self, input_data, output_data):
		# Initializer Back Propagation
		network_outputs = self.forward(input_data)
		#Calculate layer output
		layer_error = (output_data - network_outputs[-1])
		# Iterate the backward network forward
		for layer in reversed(self.layers):
			n_layer = self.layers.index(layer)
			delta = layer_error * self.sigmoid_derivate(network_outputs[n_layer])
			if(n_layer==0): #output layer
				gradient =  input_data.T.dot(delta) # output delta
			else: #hidden layers
				gradient = network_outputs[n_layer-1].T.dot(delta) #hidden delta
			layer_error = delta.dot(self.layers[n_layer].synaptic_weights.T)
			# Gradient descendent, Updates weights and bias
			layer.synaptic_weights += gradient * self.learn_rate
			layer.bias += delta.sum(axis=0) * self.learn_rate

	def train(self, training_set_inputs,
			  training_set_outputs,
			  epoch_number=False):
		epochs_control = 0
		while True:
			self.backpropagation(training_set_inputs, training_set_outputs)
			network_output = self.input(training_set_inputs)
			MSE = ((training_set_outputs - network_output) ** 2).sum()
			self.errors.append(MSE)
			# if is supervised, automatic stop on x min MSE
			if(epoch_number is False):
				if MSE < 0.009:
					return True
			else:
				if(epochs_control == epoch_number):
					return True
			epochs_control += 1

	def forward(self, input):
		output_stack = []
		for layer in self.layers:
			z = np.dot(input, layer.synaptic_weights) + layer.bias
			out = self.sigmoid_function(z)
			output_stack.append(out)
			input = out
		return output_stack

	def input(self, input):
		return self.forward(input)[-1]

	def show_error(self):
		# Imprime los errores
		plt.plot(self.errors)
		plt.show()


if __name__ == "__main__":

	# Capa 1 = capa con 2 entradas conectadas a 8neuronas que a la vez se conectan a 1 neurona de salida
	layer_model = [
		NeuronLayer(2, 6),
		NeuronLayer(6, 1)
	]

	# Se asigna el modelo de capas a la red
	neural_network = NeuralNetwork(layer_model)

	def xor_problem():
		# definicion del set de entrenamiento problema XOR
		inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
		outputs = np.array([[0],  [1],   [1],   [0]])

		# entrenamiento
		neural_network.train(inputs, outputs, 1200)
		neural_network.learn_rate = 0.01
		# test
		print(neural_network.input(inputs)) 
		neural_network.show_error()

	xor_problem()
