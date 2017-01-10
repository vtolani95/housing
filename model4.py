#.117 kaggle score
import numpy as np
import pdb
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.utils import shuffle
import sklearn.metrics as metrics
import util2 as util

def standardize_data(train, test):
  scaler = preprocessing.StandardScaler().fit(train)
  return scaler.transform(train), scaler.transform(test)

def split_data(X_train, y_train):
  cutoff = int(len(X_train) * .8)
  return X_train[:cutoff], y_train[:cutoff], X_train[cutoff:], y_train[cutoff:]

def predict(model, X):
  return np.around(np.exp(model.predict(X)), 3)

def score(y, pred, name):
  def rmsle(y, pred):
    return np.sqrt(np.average((np.log(pred+1) - np.log(y+1))**2))
  print(name + " RMSLE: %.6f R2: %.6f" % (rmsle(y, pred), metrics.r2_score(y, pred)))

def reduce_rank(X, cutoff, local_cutoff):
  U, sv, V = np.linalg.svd(X, full_matrices=0)
  sv[sv <= cutoff] = 0.0
  result = U.dot(np.diag(sv)).dot(V)
  result[result <= local_cutoff] = 0
  return result


# np.random.seed(42)
Xtrain, ytrain = util.load_data('./data/train.csv')
Xtest, test_ids = util.load_data('./data/test.csv', False)
Xtrain, ytrain = shuffle(Xtrain, ytrain.T)
# Xtrain, ytrain, Xvalid, yvalid = split_data(Xtrain, ytrain)

Xtrain = reduce_rank(Xtrain, 1e-5, .1) #cutoffs chosen via grid search cv
alphas = np.array((.1, 1, 10, 15, 20, 30, 50))
model = linear_model.RidgeCV(alphas)
model.fit(Xtrain, ytrain)
train_pred = predict(model, Xtrain)
score(np.exp(ytrain), train_pred, 'Training')
# valid_pred = predict(model, Xvalid)
# score(np.exp(yvalid), valid_pred, 'Valid')

ytest = predict(model, Xtest)
util.write_csv('model4.csv', test_ids, ytest)


