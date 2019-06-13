import csv
import numpy as np
import time
from numpy.linalg import norm 

x = []
label = []
t0=time.time()
with open('train.csv','r') as f:
	next(f)	#ignora o cabeçalho
	reader = csv.reader(f, delimiter = ',')
	for row in reader: 
		x.append([int(i) for i in row[1:]]) #desconsidera a coluna de label
		label.append(int(row[0]))	#Insere os labels nesta lista


x = [val/norm(row) for row in x for val in row]  #[val/255 for row in x for val in row] 
t1=time.time()
print(t1-t0)
