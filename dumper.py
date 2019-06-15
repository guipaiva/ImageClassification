import csv
import pickle
import numpy as np
from numpy import linalg as la

#Vetor de normas
norms=[]

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

#cria um vetor com as normas de cada vetor imagem
for row in x:
	norms.append(la.norm(row))

#normaliza o vetor dividindo os elementos pela norma
for i in range(len(x)):
	x[i] = x[i]/norms[i]

#salva os dados em formato binário para agilizar os processos de leitura dos dados
with open("dump.pkl","bw") as dmp:
	data = (x, labels, norms)
	pickle.dump(data, dmp)
