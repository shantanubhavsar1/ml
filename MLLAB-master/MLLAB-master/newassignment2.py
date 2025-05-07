#Importing all the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Supress Warnings

#Read the given CSV file, and view some sample records
advertising = pd.read_csv( "advertising.csv" )
advertising.head()

#Importing seaborn library for visualizations
import seaborn as sns

#to plot all the scatterplots in a single plot
sns.pairplot(advertising, x_vars=[ 'TV', ' Newspaper','Radio' ], y_vars = 'Sales', size = 4, kind = 'scatter' )
plt.show()
#To plot heatmap to find out correlations
sns.heamap( advertising.corr(), cmap = 'YlGnBl', annot = True )
plt.show()
#To plot heatmap to find out correlations
sns.heamap( advertising.corr(), cmap = 'YlGnBl', annot = True )
plt.show()

X = advertising[ 'TV' ]
y = advertising[ 'Sales' ]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = 0.7, test_size = 0.3, random_state = 100 )

print( X_train.shape )
print( X_test.shape )
print( y_train.shape )
print( y_test.shape )

import statsmodels.api as sm

# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)
# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

# Print the parameters,i.e. intercept and slope of the regression line obtained
lr.params

#Performing a summary operation lists out all different parameters of the regression line fitted
print(lr.summary())

# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)
# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)

#Imporitng libraries
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#RMSE value
print( "RMSE: ",np.sqrt( mean_squared_error( y_test, y_pred ) ))
#R-squared value
print( "R-squared: ",r2_score( y_test, y_pred ) )

X_train_lm = X_train_lm.values.reshape(-1,1)
X_test_lm = X_test_lm.values.reshape(-1,1)

print(X_train_lm.shape)
print(X_train_lm.shape)

from sklearn.linear_model import LinearRegression
#Representing LinearRegression as lr (creating LinearRegression object)
lr = LinearRegression()
#Fit the model using lr.fit()
lr.fit( X_train_lm , y_train_lm )

#get intercept
print( lr.intercept_ )
#get slope
print( lr.coef_ )