import pickle 

#carrega o arquivo binário criado e atribui às variáveis
with open("train_dump.pkl","br") as dmp:
	data = pickle.load(dmp)
train_x = data[0]
train_labels = data[1]
sizes = data[2]
