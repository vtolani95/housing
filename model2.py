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

def load_data(filename, train=True):
  with open(filename, newline='') as csvfile:
    x = []
    y = []
    reader = csv.reader(csvfile)
    flag = 0
    for row in reader:
      if flag == 0:
        flag = 1
        continue
      if train:
        y.append(np.log(float(row[-1])))
      else:
        y.append(int(row[0]))
      features = []
      util.lot_frontage(features, row)
      util.lot_area(features, row)
      util.street(features, row)
      util.alley(features, row)
      util.lot_shape(features, row)
      util.land_contour(features, row)
      util.utilities(features, row)
      util.lot_config(features, row)
      util.land_slope(features, row)
      util.neighborhood(features, row)
      util.condition(features, row)
      util.building_type(features, row)
      util.house_style(features, row)
      util.overall_qual(features, row)
      util.overall_cond(features, row)
      util.roof_style(features, row)
      util.roof_material(features, row)
      util.exterior_covering(features, row)
      util.mas_vnr_type(features, row)
      # util.mas_vnr_area(features, row)
      util.exterior_qual(features, row)
      util.exterior_cond(features, row)
      util.foundation(features, row)
      util.basement_qual(features, row)
      util.basement_cond(features, row)
      util.basement_exposure(features, row)
      util.basement_total_sq_ft(features, row)
      util.heating(features, row)
      # util.heating_qc(features, row)
      util.central_air(features, row)
      util.electrical_system(features, row)
      # util.low_quality_fin_sf(features, row)
      util.gr_live_area(features, row)
      util.bsmnt_full_bath(features, row)
      # util.bsmnt_half_bath(features, row)
      util.full_bath(features, row)
      util.half_bath(features, row)
      util.bedroom_abv_grd(features, row)
      util.kitchen_abv_grd(features, row)
      util.kitchen_quality(features, row)
      util.total_rooms_abv_grd(features, row)
      util.functionality(features, row)
      util.fireplaces(features, row)
      util.fireplace_quality(features, row)
      util.garage_type(features, row)
      util.garage_finish(features, row)
      util.garage_cars(features, row)
      util.garage_area(features, row)
      util.garage_quality(features, row)
      util.garage_cond(features, row)
      util.paved_driveway(features, row)
      util.wood_deck_area(features, row)
      # util.open_porch_sf(features, row)
      util.encolosed_porch(features, row)
      util.three_season_porch(features, row)
      util.screen_porch(features, row)
      util.pool_area(features, row)
      util.pool_quality(features, row)
      util.fence_quality(features, row)
      util.misc_val(features, row)
      util.sale_type(features, row)
      util.sale_condition(features, row)



      x.append(features)
  return np.array(x), np.array(y)

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

Xtrain, ytrain = load_data('./data/train.csv')
Xtest, test_ids = load_data('./data/test.csv', False)
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


