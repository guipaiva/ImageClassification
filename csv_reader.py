import csv
import numpy as np
import time
from numpy.linalg import norm 

#Matriz de entrada
x = []

#Vetor de Saída esperado
label = []

with open('train.csv','r') as f:
	#ignora o cabeçalho
	next(f)	
	reader = csv.reader(f, delimiter = ',')
	for row in reader: 
		#insere os valores em x desconsiderando a coluna de label
		x.append([int(i) for i in row[1:]])
		#Insere os labels nesta lista 
		label.append(int(row[0]))	

#normaliza os dados de entrada
#Divide por 255 se valor != 0
x =[[val/255 if val!= 0 else 0 for val in row] for row in x] 
#x = [[val/norm(row) if val!=0 else 0 for val in row ] for row in x]