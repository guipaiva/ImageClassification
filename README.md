# Image Classification
Classificando imagens utilizando Redes Neurais Artificais em Python.

## Descrição
Este programa classifica imagens de números manuscritos do dataset MNIST utilizando uma rede neural artificial a partir de imagens pré-processadas em um arquivo csv, onde cada linha corresponde aos 784 píxeis de uma imagem 28x28, com valores entre 0 e 255 em uma escala de cinza. 

## Instruções
Para a execução siga as instruções detalhadas abaixo.

* Descompactar o [arquivo compactado](Data.zip).
* Instalar os módulos
* Serializar os dados
* Executar o programa

### Instalação
Utilize o pacote [pip](https://pypi.org/project/pip/) para instalar os módulos necessários.

* [Matplotlib](https://matplotlib.org/)

* [Tensorflow](https://www.tensorflow.org/)

* [Keras](https://keras.io/)

```bash
pip3 install matplotlib
pip3 install tensorflow
pip3 install keras
```

### Serialização
Para facilitar a leitura dos dados do csv, foi utilizada a biblioteca pickle para serializar os dados em um arquivo _dump_ binário. 
```bash
python3 test_dumper.py
python3 train_dumper.py
```
### Execução
```bash
python3 main.py
```
