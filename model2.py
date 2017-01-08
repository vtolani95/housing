import numpy as np
import pdb
import csv
from sklearn import linear_model
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
  print(name + " RMS: {0}".format(metrics.mean_squared_error(y, pred)))
  print(name + " R2: {0}".format(metrics.r2_score(y, pred)))

def write_csv(filename, id, pred):
  c = csv.writer(open(filename, "wt"))
  c.writerow(['Id', 'SalePrice'])
  for i in range(len(pred)):
    c.writerow((id[i], pred[i]))

Xtrain, ytrain = util.load_data('./data/train.csv')
Xtest, test_ids = util.load_data('./data/test.csv', False)
# Xtrain, Xtest = standardize_data(Xtrain, Xtest)
# Xtrain, ytrain = shuffle(Xtrain, ytrain.T)
# Xtrain, ytrain, Xvalid, yvalid = split_data(Xtrain, ytrain)
reg = linear_model.LinearRegression()
reg.fit(Xtrain, ytrain)
train_pred = predict(reg, Xtrain)
# valid_pred = predict(reg, Xvalid)
score(np.exp(ytrain), train_pred, 'Training')
# score(np.exp(yvalid), valid_pred, 'Validation')

ytest = predict(reg, Xtest)
write_csv('model2.csv', test_ids, ytest)


