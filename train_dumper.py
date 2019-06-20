import csv
import pickle
import numpy as np
from numpy import linalg as la
import time

t0 = time.time()
#Vetor de normas
norms=[]

'''lista com o número de neurônios em cada camada
		sizes[0] = camada de input, sizes[1] = camada oculta
		sizes[2] = camada de output'''
sizes = [784,50,10]
with open('train.csv','r') as f:
	reader = csv.reader(f, delimiter = ',')
	#ignora o cabeçalho
	next(f)	
	#lê o arquivo como lista e converte para numpy array 
	data = np.array(list(reader),dtype='float')
	#remove o label de data, coloca os dados dos pixels em 'x' e os labels no vetor 'labels'
	x = data[:, 1:]
	labels = data[:, :1]
labels = labels.flatten()

'''transforma o label na representação "one-hot"
	---Não utilizado pois a biblioteca já faz a conversão---
	Ex: labels = 1, y= [0,1,0,0,0,0,0,0,0,0]
	labels = np.eye(10)[labels.astype(int)]
'''

#cria um vetor com as normas de cada vetor imagem
for row in x:
	norms.append(la.norm(row))

'''normaliza o vetor dividindo os elementos pela norma 
---- Não utilizado pois gerou imprecisão no treinamento da rede ----
for i in range(len(x)):
	x[i] = x[i]/norms[i]
'''

x/=255.0

#salva os dados em formato binário para agilizar os processos de leitura dos dados
with open("train_dump.pkl","bw") as dmp:
	data = (x, labels, sizes)
	pickle.dump(data, dmp)
t1 = time.time()
print('Os dados foram lidos e preparados, tempo total: {:.2f}s'.format(t1-t0))