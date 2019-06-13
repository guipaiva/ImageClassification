from csv_reader import *

class network(object):
	def __init__(self, sizes):
		#quantidade de camadas, incluindo entrada e saída
		self.layers = len(sizes) 
		#lista com o número de neurônios em cada camada
		self.sizes = sizes 
		#Bias não é definido para camada de entrada, por isso começa da posição 1
		self.bias = [np.random.rand(y) for y in sizes[1:]] 
		#a matriz de pesos theta e a combinação de todos os neurônios de uma camada com os da camada seguinte
		self.theta = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
		
	#Calcula a função sigmoide com theta transposto
	def sigmoid(theta, x):					
		return 1/1+exp(-theta.T.dot(x))		

	#Derivada da função sigmoide
	def der_sigmoid(theta, x):
		return sigmoid(theta, x)* (1 - sigmoid(theta, x))	

	#Faz a combinação dos bias com os thetas dada a entrada x
	def forward_propagation(theta, x):
		for b, t in zip(self.bias, self.theta): 
			h = sigmoid(theta,x)+b
			return h