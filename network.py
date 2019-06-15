from loader import *
import numpy as np

'''
	- o bias e os pesos (matriz theta) são gerados de modo aleatório aqui
'''
class network(object):
	def __init__(self, sizes):
		#quantidade de camadas, incluindo entrada e saída
		self.layers = len(sizes)
		#lista com o número de neurônios em cada camada
		#Exemplo: sizes = [784, 15, 10], sizes[0] = camada de input, sizes[1] = camada oculta
		#sizes[2] = camada de output
		self.sizes = sizes
		#Bias não é definido para camada de entrada, por isso começa da posição 1
		self.bias = [np.random.rand(y) for y in sizes[1:]]
		print(self.bias)
		#Matriz de pesos theta , com a conexão de todos os neurônios de uma camada com os da camada seguinte
		#Considerando a matriz sizes = [784,15,10], theta[0] = matriz[784x15]; theta[1] = matriz [15x10]
		self.theta = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

	#Calcula a função sigmoide com theta transposto (função de ativação)
	def sigmoid(theta, x):
		return 1/(1+exp(-theta.T.dot(x)))

	#Derivada da função sigmoide
	def der_sigmoid(theta, x):
		return sigmoid(theta, x)* (1 - sigmoid(theta, x))

	#Faz a combinação dos bias com os thetas dada a entrada x
	def forward_propagation(theta, x):
		for b, t in zip(self.bias, self.theta):
			h = sigmoid(t,x)+b
		#hipotese de saída do neuronio
		return h 

	def coust_function(layers,sizes,theta,x,y,v_lambda):
		#transforma o label na representação do vetor de saída
		#Ex: labels[0] = 1, y[0]= [0,1,0,0,0,0,0,0,0,0]
		y = np.eye(10)[labels.astype(int)]
		j=0
		m = len(sizes[0]) #quantidade de neurônios de entrada
		h = 0#calcular o valor de h aqui h=aj^(i) ---------------- Criar backpropagation
		theta1 = theta[0] #Aqui eu pego apenas os pesos da camada de entrada -------------- reshape
		theta2 = theta[1] #Aqui eu pego apenas os pesos da camada oculta ------------- reshape
		for i in range(m):
        	J += -np.sum(np.multiply(y[i,:], np.log(h[i,:])) - np.multiply((1 - y[i,:]), np.log(1 - h[i,:])))/m
    	'''
			Abaixo colocamos  o termo de regularização da função, que é dado por:
			j= j + lambda/2m * {Somatório de 1  até L-1[Somatório de 1 até Sl(Somatório de 1 até Sl+1 (theta^2))]}
			L = número de camadas, por esse motivo abaixo temos 2 somatórios, temos uma camada interna e uma oculta
			Sl= número de neurônios da camada l.
		'''
    	J += (v_lambda / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
		return j