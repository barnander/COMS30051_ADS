
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Load the dataset
merged_data = pd.read_csv('merged_data.csv')


# Assuming 'merged_data' is loaded
merged_data['log_SIPRI_TIV'] = np.log1p(merged_data['SIPRI TIV for total order'])

# Plot to visualize the transformation
sns.histplot(merged_data['log_SIPRI_TIV'], kde=True)
plt.title('Log Transformed Distribution of Arms Imports')
plt.show()

# Use in a model
X = merged_data[['log_SIPRI_TIV', 'gdp', 'population']]  # Adjusted for transformed data
y = merged_data['No_Incidents']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('R-squared:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))


from scipy import stats

# Box-Cox requires all positive values. Ensure there are no zero or negative values.
merged_data['SIPRI_TIV_positive'] = merged_data['SIPRI TIV for total order'] + 1  # Shift data if necessary
merged_data['boxcox_SIPRI_TIV'], fitted_lambda = stats.boxcox(merged_data['SIPRI_TIV_positive'])

# Plot to visualize the transformation
sns.histplot(merged_data['boxcox_SIPRI_TIV'], kde=True)
plt.title('Box-Cox Transformed Distribution of Arms Imports')
plt.show()

# Use in a model
X = merged_data[['boxcox_SIPRI_TIV', 'gdp', 'population']]  # Adjusted for transformed data
y = merged_data['No_Incidents']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('R-squared:', r2_score(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
