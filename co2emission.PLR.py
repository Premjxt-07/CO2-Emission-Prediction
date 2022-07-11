

# DEPENDENCIES

import numpy as np
import pandas as pd
import pylab as py
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("FuelConsumptionCo2.csv")
# print(df.head())

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_HWY',
          'FUELCONSUMPTION_CITY', 'CO2EMISSIONS']]
# print(cdf.head())

mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]

clf= linear_model.LinearRegression()

X_train = np.asanyarray(train[['ENGINESIZE']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])

X_test = np.asanyarray(test[['ENGINESIZE']])
y_test =np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
X_train_pol = poly.fit_transform(X_train)

clf.fit(X_train_pol,y_train)

from sklearn.metrics import r2_score

test_x_poly = poly.fit_transform(X_test)
test_y_ = clf.predict(test_x_poly)

# print ('Coefficients: ', clf.coef_)
# print ('Intercept: ',clf.intercept_)

""" VISUALIZATION OF THE CURVE """

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , y_test) )

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()