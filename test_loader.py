import pickle 
#carrega o arquivo binário criado e atribui às variáveis
with open("test_dump.pkl","br") as dmp:
	data = pickle.load(dmp)
test_x = data

