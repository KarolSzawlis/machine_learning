# Multiple Linear Regression



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



dataset = pd.read_csv('50_Startups.csv')

print(dataset.head())

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,4].values



# Encoding caterogical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:,3] = labelencoder_X.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features=[3])

X = onehotencoder.fit_transform(X).toarray()



#Avoiding dummy variable trap

X = X[:,1:]



# Splitting the dataset into test and training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



# Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predict the Test set results

y_pred = regressor.predict(X_test)



#Building the optimal model using Backward Elimination

import statsmodels.formula.api as sm

X = np.append(np.ones((50,1)).astype(int),values=X, axis=1)

X_opt = X[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X[:,[0,1,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X[:,[0,3,4,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())

X_opt = X[:,[0,3,5]]

regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()

print(regressor_OLS.summary())



# Alternative solution

def backwardElimination(x, sl):

    numVars = len(x[0])

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(y, x).fit()

        maxVar = max(regressor_OLS.pvalues).astype(float)

        if maxVar > sl:

            for j in range(0, numVars - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    x = np.delete(x, j, 1)

    regressor_OLS.summary()

    return x



SL = 0.05

X_opt = X[:, [0, 1, 2, 3, 4, 5]]

X_Modeled = backwardElimination(X_opt, SL) # = X_Opt o ile usuniemy marketing spend
