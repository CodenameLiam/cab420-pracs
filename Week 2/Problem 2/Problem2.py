# numpy handles pretty much anything that is a number/vector/matrix/array
import numpy as np
# pandas handles dataframes (exactly the same as tables in Matlab)
import pandas as pd
# matplotlib emulates Matlabs plotting functionality
import matplotlib.pyplot as plt
# stats models is a package that is going to perform the regression analysis
from statsmodels import api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
# os allows us to manipulate variables on out local machine, such as paths and environment variables
import os
# self explainatory, dates and times
from datetime import datetime, date
# a helper package to help us iterate over objects
import itertools

# -------------------------------------------------------------------------------------------
# Read data
# -------------------------------------------------------------------------------------------

combined = pd.read_csv('Week 2\\Problem 1\\final.csv')
combined = combined.drop('Unnamed: 0', axis=1)


# Convert strings to datetime
combined["Date"] = pd.to_datetime(combined["Date"])

train = combined[combined.Date < datetime(year=2017, month=1, day=1)]
val = combined[((combined.Date >= datetime(year=2017, month=1, day=1)) &
                        (combined.Date < datetime(year=2018, month=1, day=1)))]
test = combined[((combined.Date >= datetime(year=2018, month=1, day=1)) &
                          (combined.Date < datetime(year=2019, month=1, day=1)))]
print('num train = {}'.format(train.shape[0]))
print('val train = {}'.format(val.shape[0]))
print('test train = {}'.format(test.shape[0]))



# -------------------------------------------------------------------------------------------
# Training simple model
# -------------------------------------------------------------------------------------------

X_bom = ['Rainfall amount (millimetres)',
         'Daily global solar exposure (MJ/m*m)',
         'Maximum temperature (Degree C)']
# want to use all variables cyclist inbound variables
X_bcc = [x for x in train.columns.values if 'Cyclists Inbound' in x]
# remove the response variable from here
X_bcc.remove('Bicentennial Bikeway Cyclists Inbound')
# combine this list of variables together by just extending the
# BOM data with the BCC data
X_variables_linear = X_bom + X_bcc
Y_variable = 'Bicentennial Bikeway Cyclists Inbound'

Y_train = np.array(train[Y_variable], dtype=np.float64)
X_train = np.array(train[X_variables_linear], dtype=np.float64)
# want to add a constant to the model (the y-axis intercept)
X_train_constant = sm.add_constant(X_train)
# also creating validation data
Y_val = np.array(val[Y_variable], dtype=np.float64)
X_val = np.array(val[X_variables_linear], dtype=np.float64)
X_val_constant = sm.add_constant(X_val)
# will create our test data as well, we will need it later on
Y_test = np.array(test[Y_variable], dtype=np.float64)
X_test = np.array(test[X_variables_linear], dtype=np.float64)
# create the linear model
model = sm.OLS(Y_train, X_train_constant)
# fit the model without any regularisation
model_1_fit = model.fit()
pred = model_1_fit.predict(X_val_constant)
pred_train = model_1_fit.predict(X_train_constant)
print('Linear Model Validation Data: RMSE = {}'.format(
  np.sqrt(mean_squared_error(Y_val, pred))))
print('Linear Model Training Data: RMSE = {}'.format(
  np.sqrt(mean_squared_error(Y_train, pred_train))))
print(model_1_fit.summary())
print(model_1_fit.params)
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_1_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Linear Regression')
# plt.show()


# -------------------------------------------------------------------------------------------
# Add quadratic terms
# -------------------------------------------------------------------------------------------

def x2fx(x, model='quadratic'):
  """Will create a quadratic, interaction and linear
  versions of our variable of interest

  Args:
    x (np.array):
      Data set with columns as features (variables) that we 
      want to generate higher order terms for
    model (str):
      Determine linear, interaction, quadratic, purequadratic terms
 
  Returns:
    Array with higher order terms added as additional columns
  """
  from itertools import combinations as comb
  linear = np.c_[np.ones(x.shape[0]), x]
  if model == 'linear':
    return linear
  if model == 'purequadratic':
    return np.c_[linear, x**2]

  interaction = np.vstack([x[:,i]*x[:,j] for i, j in
                           comb(range(x.shape[1]), 2)]).T
  print(interaction.shape)
  if model == 'interaction':
    return np.c_[linear, interaction]
  if model == 'quadratic':
    return np.c_[linear, interaction, x**2]


print(X_train.shape)
print(X_test.shape)
X_train_complex = x2fx(X_train)
X_val_complex = x2fx(X_val)
X_test_complex = x2fx(X_test)
print(X_train_complex.shape)
print(X_val_complex.shape)


# -------------------------------------------------------------------------------------------
# Training complex model
# -------------------------------------------------------------------------------------------
model = sm.OLS(Y_train, X_train_complex)
# fit the model without any regularisation
model_2_fit = model.fit()
pred_train = model_2_fit.predict(X_train_complex)
pred = model_2_fit.predict(X_val_complex)
print('Model 2 Training Data: Quadratic Variables RMSE = {}'.format(
  np.sqrt(mean_squared_error(Y_train, pred_train))))
print('Model 2 Validation Data: Quadratic Variables RMSE = {}'.format(
  np.sqrt(mean_squared_error(Y_val, pred))))
print(model_2_fit.summary())
print(model_2_fit.params)
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_2_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Model 2: Quadratic Variables')
# plt.show()

# Plot residuals
fig, ax = plt.subplots(figsize=(8,6))
plt.scatter(pred, Y_val - pred)
plt.title('Residual plot for unregularised model')
plt.xlabel('Predicted Quantity')
plt.ylabel('Residuals')


# -------------------------------------------------------------------------------------------
# Standardising data set
# -------------------------------------------------------------------------------------------

def standardise(data):
  """ Standardise/Normalise data to have zero mean and unit variance

  Args:
    data (np.array):
      data we want to standardise (usually covariates)

    Returns:
      Standardised data, mean of data, standard deviation of data
  """
  mu = np.mean(data, axis=0)
  sigma = np.std(data, axis=0)
  scaled = (data - mu) / sigma
  return scaled, mu, sigma


X_train_std, mu_train_x, sigma_train_x = standardise(X_train)
Y_train_std, mu_train_y, sigma_train_y = standardise(Y_train)
X_val_std = (X_val - mu_train_x)/sigma_train_x
Y_val_std = (Y_val - mu_train_y)/sigma_train_y
X_test_std = (X_test - mu_train_x)/sigma_train_x
Y_test_std = (Y_test - mu_train_y)/sigma_train_y
print(mu_train_x)

# remove constant term - this is not needed with standardised data, and as a constant has a std.dev of 
# 0 will result is numeric issues if we leave it in
X_train_complex = np.delete(X_train_complex, 0, 1)
X_val_complex = np.delete(X_val_complex, 0, 1)
X_test_complex = np.delete(X_test_complex, 0, 1)
X_train_complex_std, mu_train_complex_x, sigma_train_complex_x = standardise(X_train_complex)
X_val_complex_std = (X_val_complex - mu_train_complex_x)/sigma_train_complex_x
X_test_complex_std = (X_test_complex - mu_train_complex_x)/sigma_train_complex_x


# -------------------------------------------------------------------------------------------
# Apply regression
# -------------------------------------------------------------------------------------------

def rmse(actual, pred):
  return np.sqrt(mean_squared_error(actual, pred))

def r_squared(actual, predicted):
  r2 = r2_score(actual, predicted)
  return r2

def adj_r2(actual, predicted, n, p):
  r2 = r2_score(actual, predicted)
  adjr2 = 1 - (1 - r2) * (n - 1) / (n - p - 1);
  return adjr2

def evaluate_regularisation(x_train, y_train, x_val, y_val, x_test, y_test,
                            response_mu, response_sigma, alpha_list, L1_L2):
  """
  Evaluates the efficacy of regularisation for a linear model.
  
  Identifies which values regression coffections and values of alpha
  (the hyperparam for the strength of regularisation) offers the
  best performance on the validation set.

  Evaluation is required to be performed on the standardised data, to allow
  for ease of comparison (where standardised here refers to normalisation,
  such that the data has a mean of zero and a std. of one.) This will show
  
  Will display the coefficients used in the NORMALISED/STANDARDISED model 
  used to achieve the best results.

  Is able to evaluate both Ridge and Lasso regularisation (and Elasticnet
  really if you want to try it, but in this class we are sticking to Ridge
  and Lasso).

  Args:
    x_train (np.array):
      normalised predictor variable training data
    y_train (np.array):
      normalised response variable training data
    x_val (np.array):
      normalised predictor variable validation data
    y_val (np.array):
      normalised response variable validation data
    x_test (np.array):
      normalised predictor variable test data
    y_test (np.array):
      normalised response variable test data
    response_mu (np.array):
      the mean value of the response variable from the TRAINING data
    response_sigma (np.array):
      the standard deviation  of the response variable from the TRAINING data
    alpha_list (list[np.float]):
      proposed values for alpha (the regularisation hyper param, also called
      lambda in other texts and in the lectures). Each value must be greater
      than zero.
    L1_L2 (np.int):
      Boolean to say whether we want to perform Ridge or Lasso regularisation.
      When zero, will be Ridge, When one, will be Lasso.
      Note: this value can actually be a float between zero and one as well 
      if you want to try Elasticnet regression, but here in this class would
      recommend sticking to just Ridge and Lasso.

  Retuns:
    NA
  """
  # Ridge: L1_L2 = 0
  # Lasso: L1_L2 = 1
  # create the model
  model = sm.OLS(y_train, x_train)
  # initialise the value for best RMSE that is obnoxiously large, as we want this be 
  # overwritten each time RMSE is smaller, since smaller is better and we want to 
  # update our best models each time the RMSE is smaller.
  best_rmse = 10e12
  best_alpha = []
  rmse_val = []
  rmse_train = []
  best_coeffs = []
  for alpha in alpha_list:
    model_cross_fit = model.fit_regularized(alpha=alpha, L1_wt=L1_L2)
    train_pred = model_cross_fit.predict(x_train)
    val_pred = model_cross_fit.predict(x_val)
    # want to append the rmse value to a list, as will plot all values later on
    rmse_train.append(np.sqrt(mean_squared_error(y_train, train_pred)))
    rmse_val.append(np.sqrt(mean_squared_error(y_val, val_pred)))
    # if this is the model with the lowest RMSE, lets save it
    # the [-1] index says get the last value from the list (which is the most recent RMSE)
    if rmse_val[-1] < best_rmse:
      best_rmse = rmse_val[-1]
      best_alpha = alpha
      best_coeffs = model_cross_fit.params
      
  print('Best values on Validation Data set')
  # extract the gradient and the bias from the coefficients
  # The reshape will make sure the slope is a column vector
  slope = np.array(best_coeffs[0:]).reshape(-1, 1)
  # the intercept coefficient is the last index variable, which was included with the
  # sm.add_constant() method
  # use the @ operator to perform vector/matrix multiplication
  pred_val_rescaled = (x_val @ slope) * response_sigma + response_mu
  pred_train_rescaled = (x_train @ slope) * response_sigma + response_mu
  best_r2 = r_squared(y_train * response_sigma + response_mu, pred_train_rescaled)
  best_adj_r2 = adj_r2(y_train * response_sigma + response_mu, pred_train_rescaled,
                           x_train.shape[0], x_train.shape[1]) 
  best_val_rmse = np.sqrt(mean_squared_error(y_val* response_sigma + response_mu, pred_val_rescaled))
  print('Best R Squared = {}'.format(best_r2))
  print('Best Adjusted = {}'.format(best_adj_r2))
  print('Best RMSE (val) = {}'.format(best_val_rmse))
  print('Best coefficients on the normalised model')
  print('Best slope = {}'.format(slope))
  
  # now plotting some data
  fig, axs = plt.subplots(4, figsize=(20, 25))
  # plot the first values of alpha vs RMSE for train and validation data    
  axs[0].plot(np.array(alpha_list), rmse_train)
  axs[0].plot(np.array(alpha_list), rmse_val)
  axs[0].legend(['Training', 'Validation'])
  axs[0].set_title('RMSE vs Lambda')
  axs[0].set_xlabel('Lambda')
  axs[0].set_ylabel('RMSE')    
  # plot prediction and true values for test set
  axs[1].plot((y_test*response_sigma + response_mu))
  axs[1].plot((x_test @ slope) * response_sigma + response_mu)
  axs[1].legend(['Actual', 'Predicted'])
  axs[1].set_title('Test Set Performance')
  # plotting the Q-Q plot
  train_pred = (x_train @ slope).reshape(y_train.shape)
  resid = y_train - train_pred
  sm.qqplot(resid, ax=axs[2], line='s')
  axs[2].set_title('Q-Q Plot for Linear Regression')
  # plot the residuals as well
  axs[3].scatter(train_pred, resid)
  axs[3].set_title('Residuals for training set')
  axs[3].set_xlabel('Predicted')
  axs[3].set_ylabel('Residuals')



# -------------------------------------------------------------------------------------------
# Ridge regression
# -------------------------------------------------------------------------------------------

# Simple data
print('\n\nEvualting ridge regression performance on simple data')
alpha_list = np.linspace(0, 10.0, 1000)
evaluate_regularisation(X_train_std, Y_train_std, X_val_std, Y_val_std, X_test_std, Y_test_std,
                        mu_train_y, sigma_train_y, alpha_list, 0)

# Complete data
print('\n\nEvualting ridge regression performance on complex data')
alpha_list = np.linspace(0, 10.0, 1000)
evaluate_regularisation(X_train_complex_std, Y_train_std, X_val_complex_std, Y_val_std, X_test_complex_std, 
                        Y_test_std, mu_train_y, sigma_train_y, alpha_list, 0)

# -------------------------------------------------------------------------------------------
# Lasso regression
# -------------------------------------------------------------------------------------------

# Simple data
print('\n\nEvualting lasso regression performance on simple data')
alpha_list = np.linspace(0, 10.0, 1000)
evaluate_regularisation(X_train_std, Y_train_std, X_val_std, Y_val_std, X_test_std, Y_test_std,
                        mu_train_y, sigma_train_y, alpha_list, 1)

# Complete data
print('\n\nEvualting lasso regression performance on complex data')
alpha_list = np.linspace(0, 10.0, 1000)
evaluate_regularisation(X_train_complex_std, Y_train_std, X_val_complex_std, Y_val_std, X_test_complex_std, 
                        Y_test_std, mu_train_y, sigma_train_y, alpha_list, 1)

plt.show()