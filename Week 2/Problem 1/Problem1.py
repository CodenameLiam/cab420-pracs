from itertools import combinations as comb
from sklearn.metrics import mean_squared_error
from datetime import datetime
from statsmodels import api as sm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def x2fx(x, model='quadratic'):
    linear = np.c_[np.ones(x.shape[0]), x]

    if model == 'linear':
        return linear
    if model == 'purequadratic':
        return np.c_[linear, x**2]

    interaction = np.vstack([x[:,i]*x[:,j] for i, j in comb(range(x.shape[1]), 2)]).T

    if model == 'interactions':
        return np.c_[linear, interaction]
    if model == 'quadratic':
        return np.c_[linear, interaction, x**2]
    
# -------------------------------------------------------------------------------------------git
# Read data
# -------------------------------------------------------------------------------------------

data = pd.read_csv('Week 2\\Problem 1\\final.csv')
data = data.drop('Unnamed: 0', axis=1)


# Convert strings to datetime
data["Date"] = pd.to_datetime(data["Date"])

# -------------------------------------------------------------------------------------------
# Create training, validation and testing sets
# -------------------------------------------------------------------------------------------

train = data[data.Date < datetime(year=2017, month=1, day=1)]
validation = data[((data.Date >= datetime(year=2017, month=1, day=1)) &
                    (data.Date < datetime(year=2018, month=1, day=1)))]
test = data[data.Date >= datetime(year=2018, month=1, day=1)]

# Store BOM data headers
X_bom = ['Rainfall amount (millimetres)',
         'Daily global solar exposure (MJ/m*m)',
         'Maximum temperature (Degree C)']

# Extract inbound cyclist values as our predictors
X_bcc = [x for x in train.columns.values if 'Cyclists Inbound' in x]
# Use Bicentennial cyclists as our response
X_bcc.remove('Bicentennial Bikeway Cyclists Inbound')

# Combine BCC and BOM variables
X_variables = X_bom + X_bcc
Y_variable = 'Bicentennial Bikeway Cyclists Inbound'
Y_train = np.array(train[Y_variable], dtype=np.float64)
X_train = np.array(train[X_variables], dtype=np.float64)

# Add constant to the model
X_train = sm.add_constant(X_train)

# Create validation and testing data
Y_val = np.array(validation[Y_variable], dtype=np.float64)
X_val = np.array(validation[X_variables], dtype=np.float64)
X_val = sm.add_constant(X_val)

Y_test = np.array(test[Y_variable], dtype=np.float64)
X_test = np.array(test[X_variables], dtype=np.float64)
X_test = sm.add_constant(X_test)

# -------------------------------------------------------------------------------------------
# Model 1 - Normal terms
# -------------------------------------------------------------------------------------------

# Create the linear model
model = sm.OLS(Y_train, X_train)

# Fit the model
model_1_fit = model.fit()
pred = model_1_fit.predict(X_val)

# Print results
print('\n\nModel 1 RMSE = {}'.format(np.sqrt(mean_squared_error(Y_val, model_1_fit.predict(X_val)))))
print(model_1_fit.summary())
# print(model_1_fit.params)

# Plot results
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_1_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Linear Regression Model 1')
# plt.show()

# Residual plot
fig, ax = plt.subplots(figsize=(8,6))
plt.scatter(pred, Y_val - pred)
plt.title('Residual plot for unregularised model')
plt.xlabel('Predicted Quantity')
plt.ylabel('Residuals')


# -------------------------------------------------------------------------------------------
# Model 2 - Higher terms
# -------------------------------------------------------------------------------------------

X_train_complex = x2fx(X_train)

# Add constant to the model
X_train_complex = sm.add_constant(X_train_complex)

# Create validation and testing data
X_val_complex = x2fx(X_val)
X_val_complex = sm.add_constant(X_val_complex)

X_test_complex = x2fx(X_test)
X_test_complex = sm.add_constant(X_test_complex)

# Normal shape
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# Higher order terms shape
print(X_train_complex.shape)
print(X_test_complex.shape)
print(X_val_complex.shape)

# Create the linear model with higher order terms
model_2 = sm.OLS(Y_train, X_train_complex)

# Fit the model
model_2_fit = model_2.fit()
pred = model_2_fit.predict(X_val_complex)
pred_test = model_2_fit.predict(X_test_complex)
pred_train = model_2_fit.predict(X_train_complex)

# Print results
print('\n\nModel 2 RMSE Test = {}'.format(np.sqrt(mean_squared_error(Y_test, pred_test))))
print('\nModel 2 RMSE Train = {}'.format(np.sqrt(mean_squared_error(Y_train, pred_train))))
print('\nModel 2 RMSE Validation = {}'.format(np.sqrt(mean_squared_error(Y_val, pred))))
print(model_2_fit.summary())
# print(model_2_fit.params)

# Plot results
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_2_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Linear Regression Model 2')
# plt.show()



# -------------------------------------------------------------------------------------------
# Model 3 - Regularisation L1
# -------------------------------------------------------------------------------------------

model_3 = sm.OLS(Y_train, X_train_complex)
model_3_fit = model_3.fit_regularized(alpha=10, L1_wt=1)

pred = model_3_fit.predict(X_val_complex)

print('\n\nModel 3 L1 RMSE = {}'.format(np.sqrt(mean_squared_error(Y_val, pred))))
# print(model_3_fit.summary())
# print(model_3_fit.params)

# -------------------------------------------------------------------------------------------
# Model 4 - Regularisation L2
# -------------------------------------------------------------------------------------------

model_4 = sm.OLS(Y_train, X_train_complex)
model_4_fit = model_4.fit_regularized(alpha=10, L1_wt=0)

pred = model_4_fit.predict(X_val_complex)

print('\n\nModel 4 L2 RMSE = {}'.format(np.sqrt(mean_squared_error(Y_val, pred))))
# print(model_3_fit.summary())
# print(model_4_fit.params)

plt.show()