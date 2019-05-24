from matplotlib import pyplot as plt
import numpy as np
from csv_reader import *

def plot(x,label):
	x = np.reshape(x,(28,28))
	label = ('Label: '+str(label))
	plt.title(label)
	plt.imshow(x, cmap="gray")
	plt.show()

