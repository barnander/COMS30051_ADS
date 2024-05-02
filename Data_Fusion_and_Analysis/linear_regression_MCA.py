import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('controls.csv')

# Filter data to keep records from 1992 onwards

trade_register_data = trade_register_data[trade_register_data['Year'] >= 1992]
yearly_agg_data = yearly_agg_data[yearly_agg_data['Year'] >= 1992]
extra_data = extra_data[extra_data['Year'] >= 1992]


# Prepare and merge the data
arms_aggregated = trade_register_data.groupby(['Country', 'Year']).agg({
    'SIPRI TIV for total order': 'sum'
}).reset_index()

merged_data = pd.merge(yearly_agg_data, arms_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

# Fill missing values with 0 (assuming missing data implies no orders/incidents)
merged_data.fillna(0, inplace=True)

# Apply logarithmic transformation to 'SIPRI TIV for total order' to address skewness
merged_data['SIPRI TIV for total order_log'] = np.log1p(merged_data['SIPRI TIV for total order'])
merged_data['No_Incidents_log'] = np.log1p(merged_data['No_Incidents'])

# Prepare dependent and independent variables, include the transformed 'SIPRI TIV for total order_log'
dependent_var = merged_data['No_Incidents_log']
independent_vars = merged_data[['SIPRI TIV for total order_log', 'gdp', 'population', 'hdi', 'v2x_polyarchy']]

# Standardize the independent variables including the transformed 'SIPRI TIV for total order_log'


# Add a constant for the intercept
X = sm.add_constant(independent_vars)

# Fit the OLS model with normalized variables
model_ols = sm.OLS(dependent_var, X).fit()
print(model_ols.summary())

# Plotting the relationship using the transformed and standardized 'SIPRI TIV for total order_log'
plt.figure(figsize=(10, 6))
sns.regplot(x=independent_vars['SIPRI TIV for total order_log'], y=dependent_var, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('Log Scaled Total Value of Arms Imports (SIPRI TIV) million USD')
plt.ylabel('Log Scaled Number of Terrorist Incidents')
plt.grid(True)
plt.show()

