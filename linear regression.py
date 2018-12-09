import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Salary_Data.csv')

# print(dataset.head())

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

# Fitting Simple Linear Regression to the training set

regressor = LinearRegression()

regressor.fit(X_train, y_train)

# Predicting the test set results

y_pred = regressor.predict(X_test)

# Plots

plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train))

plt.title('Salary vs Experience (Training set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()

plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train))

plt.title('Salary vs Experience (Test set)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
