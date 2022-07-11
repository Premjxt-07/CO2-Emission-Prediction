""" 
    Python Program to predict CO2Emissions based on EngineSize . The program depicts a simple linear regression model.

    DataSet used has been downloaded from IBM Storage .
 """
# DEPENDENCIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumptionCo2.csv")
# print(df.head())  TO PRINT THE FIRST 5 rows of the dataframe used for debugging :P

""" - **MODELYEAR** e.g. 2014
- **MAKE** e.g. Acura
- **MODEL** e.g. ILX
- **VEHICLE CLASS** e.g. SUV
- **ENGINE SIZE** e.g. 4.7
- **CYLINDERS** e.g 6
- **TRANSMISSION** e.g. A6
- **FUEL CONSUMPTION in CITY(L/100 km)** e.g. 9.9
- **FUEL CONSUMPTION in HWY (L/100 km)** e.g. 8.9
- **FUEL CONSUMPTION COMB (L/100 km)** e.g. 9.2
- **CO2 EMISSIONS (g/km)** e.g. 182   --> low --> 0 """


# VISUALIZATION OF DATA

# df.describe() Is used to give a overall statistical analysis of the dataframe such as count , max , min , mean etc


# selection of particular columns from the dataframe
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head(9)

""" viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
viz.hist()
plt.show() """

""" plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()
 """
""" plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show() """

""" plt.scatter(cdf.CYLINDERS,cdf.CO2EMISSIONS,color='red')
plt.xlabel("CYLINDER")
plt.ylabel("EMISSION")
plt.show() """

# CREATING DATASET FOR ENGINESIZE VS CO2EMISSIONS

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

""" 
    np.random.rand generates a random number over uniform distribution between (0,1) . If it is less than 0.8 it returns true else false
    train = cdf[msk] selects the rows of the dataframe for which the msk value is equal to true
    test = cdf[~msk] selects the rows of the dataframe for which the msk value is equal to false (It inverts the value of msk )

"""

""" plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color ="green")
plt.xlabel("EngineSize")
plt.ylabel("CO2 Emissions")
plt.show()
 """
reg = linear_model.LinearRegression()
X_train = np.asanyarray(train[["ENGINESIZE"]])
y_train = np.asanyarray(train[["CO2EMISSIONS"]])

reg.fit(X_train, y_train)

""" print ('Coefficients: ', reg.coef_)
print ('Intercept: ',reg.intercept_)
"""

# TO PLOT THE BEST FIT LINE :

""" plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(X_train, reg.coef_[0][0]*X_train+ reg.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
"""

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_hat = reg.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat, test_y))

# R-squared is not error, but is a popular metric for accuracy of your model.
 # It represents how close the data are to the fitted regression line. 
 # The higher the R-squared, the better the model fits your data.
  # Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

plt.plot(test_y)
plt.plot(test_y_hat)
plt.show()