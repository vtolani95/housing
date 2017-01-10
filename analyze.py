import numpy as np
import pdb
import util2 as util
import matplotlib.pyplot as plt

Xtrain, ytrain = util.load_data('./data/train.csv')

for i in range(len(Xtrain.T)):
  print(i)
  plt.scatter(Xtrain[:,i], ytrain)
  plt.show()
  pdb.set_trace()

