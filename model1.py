import numpy as np
import pdb
import csv
from sklearn import linear_model


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
      # features.append(int(row[3])) #lot frontage
      features.append(int(row[4])) #lot area
      features.append(int(row[17])) #overall qual
      features.append(int(row[18])) #overall cond
      x.append(features)
  return np.array(x), np.array(y)

train, train_vals = load_data('./data/train.csv')
reg = linear_model.LinearRegression()
reg.fit(train, train_vals)
print(reg.score(train, train_vals))

test, test_ids = load_data('./data/test.csv', False)
test_pred = reg.predict(test)
c = csv.writer(open("model1.csv", "wt"))
c.writerow(['Id', 'SalePrice'])
for i in range(len(test_pred)):
  c.writerow((test_ids[i], max(np.exp(test_pred[i]), 0.0)))

