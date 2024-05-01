import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the datasets
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('/Users/jacovaneeden/Desktop/controls.csv')

# Renaming columns for consistent merging
trade_register_data.rename(columns={'Recipient': 'Country', 'Year of order': 'Year'}, inplace=True)
yearly_agg_data.rename(columns={'country_txt': 'Country', 'iyear': 'Year'}, inplace=True)
extra_data.rename(columns={'Country Name_x': 'Country', 'year': 'Year'}, inplace=True)

# Convert 'Year' to integer if not already
trade_register_data['Year'] = trade_register_data['Year'].astype(int)
yearly_agg_data['Year'] = yearly_agg_data['Year'].astype(int)
extra_data['Year'] = extra_data['Year'].astype(int)

# Aggregating the arms trade data by Country and Year
arms_aggregated = trade_register_data.groupby(['Country', 'Year']).agg({
    'SIPRI TIV for total order': 'sum'
}).reset_index()

# Merging the datasets on 'Country' and 'Year'
merged_data = pd.merge(yearly_agg_data, arms_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

# Handling missing values - filling with mean of the columns for numeric data
numeric_cols = merged_data.select_dtypes(include=[np.number]).columns.tolist()
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(merged_data.mean(numeric_only=True))

# Remove all data before 1991
merged_data = merged_data[merged_data['Year'] >= 1991]

# Ensure 'Country' and 'Year' are set as index before filtering
merged_data.set_index(['Country', 'Year'], inplace=True)

# Correct filtering for India using the .loc accessor
india_data = merged_data.loc['India']

# Define dependent and independent variables for the time-series analysis
dep_var = india_data['No_Incidents']
indep_vars = india_data[['SIPRI TIV for total order', 'gdp', 'population']]  # Specify the independent variables

# Add a constant term for the intercept to the independent variables
indep_vars = sm.add_constant(indep_vars)

# Fit the OLS model (Ordinary Least Squares)
model = sm.OLS(dep_var, indep_vars)
results = model.fit()

# Output results for India
print(results.summary())

# Plot the data points for India
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SIPRI TIV for total order', y='No_Incidents', data=india_data, color='blue', alpha=0.6, label='Data Points')
sns.lineplot(x='SIPRI TIV for total order', y=results.predict(indep_vars), data=india_data, color='red', label='Fitted Line')

plt.title('Impact of Arms Imports on Terrorist Incidents in India')
plt.xlabel('Total Value of Arms Imports (SIPRI TIV)')
plt.ylabel('Number of Terrorist Incidents')
plt.legend()
plt.grid(True)
plt.show()