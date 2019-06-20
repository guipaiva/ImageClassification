from matplotlib import pyplot as plt
import numpy as np
from test_loader import *
from train import predicts

def plot_single(x,i):
	#Desativa grid e ticks para melhor visualização da imagem
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])
	#redimensiona a matriz para 28x28
	x = np.reshape(x,(28,28))
	#define o título da imagem 
	i_title = ('Valor previsto pela rede neural: '+str(np.argmax(predicts[i]))+'\n Probabilidade: '+str(100*np.max(predicts[i])))
	plt.title(i_title,fontsize=8)
	#define o plot da imagem na escala de cinza
	plt.imshow(x, cmap="Greys") 

def plot_imgs(start):
	#número de linhas
	num_rows = 3
	#número de colunas
	num_cols = 3
	plt.figure()
	j = 1
	#cria subplots das imagens de 1 a 9 e mostra cada imagem em seu índice
	for i in range(start,start+9):  
		plt.subplot(num_rows, num_cols,j)
		plot_single(test_x[i],i)
		j+=1
	plt.show()
