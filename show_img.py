from matplotlib import pyplot as plt
from csv_reader import *

def plot(x,label):
	#redimensiona a matriz para 28x28
	x = np.reshape(x,(28,28))
	#define o título da imagem 
	label = ('Label: '+str(label)) 
	plt.title(label)
	#define o plot da imagem na escala de cinza
	plt.imshow(x, cmap="gray") 
	plt.show()

i = int(input("De 1 a 42000, qual imagem deseja exibir?\n"))

plot(x[i-1], label[i-1]) 