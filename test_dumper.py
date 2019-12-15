import csv
import pickle
import numpy as np
import time
from numpy import linalg as la

t0 = time.time()
with open('Data/test.csv','r') as f:
	#abre o arquivo como csv e define a ',' como delimitador
	reader = csv.reader(f, delimiter = ',')
	#ignora o cabeçalho
	next(f)	
	#lê o arquivo como lista e converte para numpy array 
	data = np.array(list(reader),dtype='float')
	#remove o label de data, coloca os dados dos pixels em 'x' e os labels no vetor 'labels'
	x = data

#Faz a normalização dos dados por 255 (valores entre 0 e 1)
x/=255.0

#salva os dados em formato binário para agilizar os processos de leitura dos dados
with open("test_dump.pkl","bw") as dmp:
	data = (x)
	pickle.dump(data, dmp)
t1 = time.time()

print('Os dados foram lidos e preparados, tempo total: {:.2f}s'.format(t1-t0))
