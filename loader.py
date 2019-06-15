import pickle 

#carrega o arquivo binário criado e atribui às variáveis
with open("dump.pkl","br") as dmp:
	data = pickle.load(dmp)
x = data[0]
labels = data[1]
norms = data[2]

