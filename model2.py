import numpy as np
import pdb
import csv
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
  return np.exp(model.predict(X))

def score(y, pred, name):
  def rmsle(y, pred):
    return np.sqrt(np.average((np.log(pred+1) - np.log(y+1))**2))
  print(name + " RMSLE: %.6f R2: %.6f" % (rmsle(y, pred), metrics.r2_score(y, pred)))

def write_csv(filename, id, pred):
  c = csv.writer(open(filename, "wt"))
  c.writerow(['Id', 'SalePrice'])
  for i in range(len(pred)):
    c.writerow((id[i], pred[i]))

np.random.seed(35)
Xtrain, ytrain = util.load_data('./data/train.csv')
Xtest, test_ids = util.load_data('./data/test.csv', False)

Xtrain, ytrain = shuffle(Xtrain, ytrain.T)
Xtrain, ytrain, Xvalid, yvalid = split_data(Xtrain, ytrain)
# Xtrain, Xtest = standardize_data(Xtrain, Xtest)

reg = linear_model.LinearRegression()
reg.fit(Xtrain, ytrain)
train_pred = predict(reg, Xtrain)
valid_pred = predict(reg, Xvalid)
score(np.exp(ytrain), train_pred, 'Training')
score(np.exp(yvalid), valid_pred, 'Validation')

ytest = predict(reg, Xtest)
write_csv('model2.csv', test_ids, ytest)


