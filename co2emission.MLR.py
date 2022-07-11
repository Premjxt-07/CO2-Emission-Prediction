

# DEPENDENCIES
import pandas as pd
import numpy as np
import pylab as py
import matplotlib as plt
from sklearn import linear_model


df = pd.read_csv("FuelConsumptionCo2.csv")
# print(df.head())

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB',
          'CO2EMISSIONS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY']]
# print(cdf.head())

reg = linear_model.LinearRegression()

mask = np.random.rand(len(df)) < 0.8
train = cdf[mask]
test = cdf[~mask]


# We train the model using FUELCONSUMPTION_COMB which is the combinational result of FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY and calculate the variance score

""" X_train = np.asanyarray(
    train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(X_train, y_train)

# print("Coefficients :" , reg.coef_)

X_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

y_predict = reg.predict(X_test)

print("Residual sum of squares: %.2f"
      % np.mean((y_predict - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(X_test, y_test))
 """

# We train the model using FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY individually

X_train = np.asanyarray(
    train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(X_train, y_train)

# print("Coefficients :" , reg.coef_)

X_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])

y_predict = reg.predict(X_test)

print("Residual sum of squares: %.2f"
      % np.mean((y_predict - y_test) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % reg.score(X_test, y_test))

