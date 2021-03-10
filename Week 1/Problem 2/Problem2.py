from datetime import datetime
from statsmodels import api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def clean_data():
    # Read data from P1
    data = pd.read_csv('Week 1\\Problem 2\\combined.csv')
    data['Date']

    # Remove columns with too many NaN cells
    threshold = 300

    for column in data.columns.values:
        if np.sum(data[column].isna()) > threshold:
            data = data.drop(column, axis=1)

    # Remove rows containing NaN cells
    data = data.dropna(axis=0)
    data = data.drop('Unnamed: 0', axis=1)

    # Export filtered list to CSV
    data.to_csv("Week 1\\Problem 2\\final.csv")

    return data

# Get data
try:
    data = pd.read_csv('Week 1\\Problem 2\\final.csv')
    data = data.drop('Unnamed: 0', axis=1)
except:
    data = clean_data()

# Convert strings to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Create training, validation and testing sets
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
# Model 1
# -------------------------------------------------------------------------------------------

# Create the linear model
model = sm.OLS(Y_train, X_train)

# Fit the model
model_1_fit = model.fit()
pred = model_1_fit.predict(X_val)

# Print results
print('Model 1 RMSE = {}'.format(np.sqrt(mean_squared_error(Y_val, model_1_fit.predict(X_val)))))
print(model_1_fit.summary())
print(model_1_fit.params)

# Plot results
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_1_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Linear Regression')
# plt.show()

# Plot correlation between predictors and response
all_variables = X_variables + ['Bicentennial Bikeway Cyclists Inbound']
corr_coeffs = train[all_variables].corr()
plt.figure(figsize=[15, 15])
plt.matshow(corr_coeffs)
plt.colorbar();
# plt.show()

# -------------------------------------------------------------------------------------------
# Model 2
# -------------------------------------------------------------------------------------------

# Remove uncorrelated variables
to_remove = [X_variables[0]]
print('Variables to remove -> {}'.format(to_remove[0]))
train = train.drop(X_variables[0], axis=1)
X_variables.remove(to_remove[0])
print(X_variables)

# Create new model
X_train = np.array(train[X_variables], dtype=np.float64)
X_train = sm.add_constant(X_train)

Y_val = np.array(validation[Y_variable], dtype=np.float64)
X_val = np.array(validation[X_variables], dtype=np.float64)
X_val = sm.add_constant(X_val)

Y_test = np.array(test[Y_variable], dtype=np.float64)
X_test = np.array(test[X_variables], dtype=np.float64)
X_test = sm.add_constant(X_test)

# Fit model
model_2 = sm.OLS(Y_train, X_train)
model_2_fit = model_2.fit()
pred = model_2_fit.predict(X_val)
print('Model 1 RMSE = {}'.format(np.sqrt(mean_squared_error(Y_val, model_2_fit.predict(X_val)))))
print(model_2_fit.summary())
print(model_2_fit.params)

# -------------------------------------------------------------------------------------------
# Model 3
# -------------------------------------------------------------------------------------------

# Remove uncorrelated variables
to_remove = [X_variables[3]]
print('Variable to remove -> {}'.format(to_remove[0]))
train = train.drop([X_variables[3]], axis=1)
X_variables.remove(to_remove[0])
print(X_variables)

# Create new model
X_train = np.array(train[X_variables], dtype=np.float64)
X_train = sm.add_constant(X_train)

Y_val = np.array(validation[Y_variable], dtype=np.float64)
X_val = np.array(validation[X_variables], dtype=np.float64)
X_val = sm.add_constant(X_val)

Y_test = np.array(test[Y_variable], dtype=np.float64)
X_test = np.array(test[X_variables], dtype=np.float64)
X_test = sm.add_constant(X_test)

# Fit model
model_3 = sm.OLS(Y_train, X_train)
model_3_fit = model_3.fit()
pred = model_3_fit.predict(X_val)
print('Model 1 RMSE = {}'.format(
  np.sqrt(mean_squared_error(Y_val, model_3_fit.predict(X_val)))))
print(model_3_fit.summary())
print(model_3_fit.params)

# -------------------------------------------------------------------------------------------
# Model 4
# -------------------------------------------------------------------------------------------

# Remove uncorrelated variables
to_remove = [X_variables[0]]
print('Variables to remove -> {}'.format(to_remove[0]))
train = train.drop(X_variables[0], axis=1)
X_variables.remove(to_remove[0])
print(X_variables)

# Create new model
X_train = np.array(train[X_variables], dtype=np.float64)
X_train = sm.add_constant(X_train)

Y_val = np.array(validation[Y_variable], dtype=np.float64)
X_val = np.array(validation[X_variables], dtype=np.float64)
X_val = sm.add_constant(X_val)

Y_test = np.array(test[Y_variable], dtype=np.float64)
X_test = np.array(test[X_variables], dtype=np.float64)
X_test = sm.add_constant(X_test)

# Fit the model
model_4 = sm.OLS(Y_train, X_train)
model_4_fit = model_4.fit()
pred = model_4_fit.predict(X_val)
print('Model 4 RMSE = {}'.format(np.sqrt(mean_squared_error(Y_val, model_4_fit.predict(X_val)))))
print(model_4_fit.summary())
print(model_4_fit.params)

# Plot results
fig, ax = plt.subplots(figsize=(8,6))
sm.qqplot(model_4_fit.resid, ax=ax, line='s')
plt.title('Q-Q Plot for Linear Regression')
# plt.show()

# Plot correlation between predictors and response
all_variables = X_variables + ['Bicentennial Bikeway Cyclists Inbound']
corr_coeffs = train[all_variables].corr()
plt.figure(figsize=[15, 15])
plt.matshow(corr_coeffs)
plt.colorbar();
# plt.show()

# Run model on test data
pred = model_4_fit.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(Y_test, pred))
fig = plt.figure(figsize=[12, 8])
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.arange(len(pred)), pred, label='Predicted')
ax.plot(np.arange(len(Y_test)), Y_test, label='Actual')
ax.set_title(rmse_test)
ax.legend()
plt.show()

