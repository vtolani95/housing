#.145 kaggle score
import numpy as np
import pdb
from sklearn import linear_model
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.utils import shuffle
import sklearn.metrics as metrics
import util

def standardize_data(train, test):
  scaler = preprocessing.StandardScaler().fit(train)
  return scaler.transform(train), scaler.transform(test)

def split_data(X_train, y_train):
  cutoff = int(len(X_train) * .8)
  return X_train[:cutoff], y_train[:cutoff], X_train[cutoff:], y_train[cutoff:]

def predict(model, X):
  return np.exp(model.predict(X))

def score(y, pred, name):
  # print(name + " RMS: {0}".format(metrics.mean_squared_error(y, pred)))
  print(name + " R2: {0}".format(metrics.r2_score(y, pred)))

def reduce_rank(X, cutoff, local_cutoff):
  U, sv, V = np.linalg.svd(X, full_matrices=0)
  sv[sv <= cutoff] = 0.0
  result = U.dot(np.diag(sv)).dot(V)
  result[result <= local_cutoff] = 0
  return result


np.random.seed(42)
Xtrain, ytrain = util.load_data('./data/train.csv')
Xtest, test_ids = util.load_data('./data/test.csv', False)
# Xtrain, ytrain = shuffle(Xtrain, ytrain.T)
# Xtrain, ytrain, Xvalid, yvalid = split_data(Xtrain, ytrain)

# Xtrain = reduce_rank(Xtrain, 1e-5, .1) #cutoffs chosen via grid search cv
num_trees = [70, 120, 150]
learning_rates = [.001, .1, .5, 1, 5, 10]
max_depth = [17, 20,]
# for a in alphas:
# model = ensemble.RandomForestRegressor(30, max_depth =7)
# model = tree.DecisionTreeRegressor(criterion='mse', splitter='random', max_depth=5)
# for num in num_trees:
#   # for rate in learning_rates:
#   for depth in max_depth:
#     # print (str(num) + " LR: " + str(rate))
#     # model = ensemble.AdaBoostRegressor(n_estimators=num, learning_rate=rate)
#     print(str(num) + " DEPTH: " + str(depth))
model = ensemble.RandomForestRegressor(150, max_depth =20)
model.fit(Xtrain, ytrain)
train_pred = predict(model, Xtrain)
score(np.exp(ytrain), train_pred, 'Training')
# valid_pred = predict(model, Xvalid)
# score(np.exp(yvalid), valid_pred, 'Valid')

ytest = predict(model, Xtest)
util.write_csv('model5.csv', test_ids, ytest)


