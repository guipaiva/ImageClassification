class network(object):
	def __init__(self, sizes):
		self.layers = len(sizes) #quantidade de camadas, incluindo entrada e saída
		self.sizes = sizes #lista com o número de neurônios em cada camada
		self.bias = [np.random.rand(y) for y in sizes[1:]] #Bias não é definido para camada de entrada, por isso começa da posição 1
		self.theta = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
		#a matriz de pesos theta a combinação de todos os neurônios de uma camada com os da camada seguinte

	def sigmoid(theta, x):					
		return 1/1+exp(-theta.T.dot(x))		#Calcula a função sigmoide com theta transposto

	def der_sigmoid(theta, x):
		return sigmoid(theta, x)* 1 - sigmoid(theta, x)	#Derivada da função sigmoide

	def forward_propagation(theta, x):
		for b, t in zip(self.bias, self.theta): #faz a combinação dos bias com os thetas dada a entrada x
			x = sigmoid(np.dot(t.T,x)+b)