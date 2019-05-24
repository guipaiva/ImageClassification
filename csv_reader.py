import csv

x = []
label = []
with open('train.csv','r') as f:
	next(f)	#ignora o cabeçalho
	reader = csv.reader(f, delimiter = ',')
	for row in reader:
		x.append([int(i) for i in row[1:]]) #desconsidera a coluna de label
		label.append(int(row[0]))	#Insere os labels nesta lista para consulta
