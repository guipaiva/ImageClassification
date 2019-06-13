from matplotlib import pyplot as plt
import numpy as np
from csv_reader import *

def plot(x,label):
	x = np.reshape(x,(28,28)) #redimensiona a matriz para 28x28
	label = ('Label: '+str(label)) 
	plt.title(label) #define o título da imagem
	plt.imshow(x, cmap="gray") #define o plot da imagem na escala de cinza
	plt.show()

