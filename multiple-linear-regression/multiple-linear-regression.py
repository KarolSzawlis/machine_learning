# Multiple Linear Regression



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv('insurance.csv')

#print(dataset.head())

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,6].values



# Encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X0 = LabelEncoder()

X[:,1] = labelencoder_X0.fit_transform(X[:,1])



labelencoder_X1 = LabelEncoder()

X[:,4] = labelencoder_X1.fit_transform(X[:,4])



labelencoder_X2 = LabelEncoder()

X[:,5] = labelencoder_X2.fit_transform(X[:,5])



onehotencoder = OneHotEncoder(categorical_features=(1,4,5))



X = onehotencoder.fit_transform(X).toarray()



# Avoiding dummy variable trap

X = X[:,[1,3,5,6,7,8,9,10]]



# Splitting the dataset into test and training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predict the Test Set results

y_pred = regressor.predict(X_test)



# Backward elimination (0.05)

import statsmodels.formula.api as sm

X = np.append(np.ones((1338,1)).astype(int),values=X, axis=1)

X_opt = X[:,[0,1,2,3,4,5,6,7]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



X_opt = X[:,[0,2,3,4,5,6,7]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



X_opt = X[:,[0,2,4,5,6,7]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())
