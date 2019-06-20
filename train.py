from train_loader import *
from test_loader import test_x
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

'''
    A biblioteca Keras é uma biblioteca API da biblioteca tensorflow,
    ou seja as funções empregadas por essa biblioteca funcionam como um intermédio entre a biblioteca
    citada e programadores.
'''
#A função Sequential cria o modelo da nossa rede neural de forma empilhada (o output de uma é o input da próxima camada)
model = Sequential()

#Cria uma camada com 50 neurônios(primeiro argumento), que se originaram de 784 variáveis(segundo argumento)
model.add(Dense(sizes[1], input_shape=(sizes[0],), activation="sigmoid"))

#Cria uma camada com 10 neurônios 'Lista sizes definida no algorimo train_dumper'
model.add(Dense(sizes[2], activation="sigmoid")) 

'''
    As camadas acima são ativadas pela função de ativação "sigmoid" último argumento das 2 funções acima, 
    que retornam um valor de probabilidade entre 0 e 1.
    As linhas acima geram uma rede no formato:
    784 (entrada) -> 50 (camada oculta) -> 10 (saída)
'''

model.compile(optimizer= SGD(0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''
Na linha acima compilamos nossa rede. Aqui o tensorflow automaticamente escolhe o melhor caminho para
representar a rede e fazer predições. Foi usado na função acima
    - Otimizador(atualizar o modelo com base nos dados e na função de custo): SGD - Stochastic Gradient Descent- com alpha 0.01 ;
    - Perda (mede quão bem o modelo classifica durante o treino, afim de minimizar os erros e ajustar o modelo): 'sparse_categorical_crossentropy'  (segundo argumento),
    - Métrica (usada para monitorar os passos de treino e teste): accuracy (cálcula a precisão - razão - da correspondência entre o valor real e sua predição)
'''

model.fit(train_x,train_labels,epochs=5)
'''
    Na linha acima, treinamos de fato nossa rede neural. Para tanto, fazemos o seguinte:
        - Alimentamos o modelo com os dois primeiros argumentos da função.
        - O modelo associa os pixels das imagens a seus respectivos valores das imagens
        - Previsão entre argumentos acima é montada.
        - O último argumento é o número de iterações (neste caso, 5)
    Ao fim a função retorna um histórico das métricas do algoritmo, o que possibilita a verificação
    da precisão de nosso algoritmo.
'''
predicts = model.predict(test_x)
'''
    A lista acima é gerada após o treinamento da rede neural, com a predição para os dados de teste
    fornecidos no arquivo test.csv e processados pelo algoritmo test_dumper
'''