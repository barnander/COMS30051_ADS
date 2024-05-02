import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
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

# Ensure that 'Country' and 'Year' are set as index
merged_data.set_index(['Country', 'Year'], inplace=True)

# Filter for India
india_data = merged_data.xs('India', level='Country')

# Filter for India
india_data = merged_data[merged_data.index.get_level_values(0) == 'India']


# Define dependent and independent variables without adding a constant
dep_var = india_data['No_Incidents']
indep_vars = india_data[['SIPRI TIV for total order', 'gdp', 'population']]  # Variables

# Initialize PanelOLS model for India with check_rank=False to bypass the rank check
mod = PanelOLS(dep_var, indep_vars, entity_effects=True, check_rank=False)

# Fit the model with robust covariance estimation for India
results = mod.fit(cov_type='robust')

# Output results for India
print(results)

# Plot the data points for India
plt.figure(figsize=(10, 6))
sns.scatterplot(x='SIPRI TIV for total order', y='No_Incidents', data=india_data, color='blue', alpha=0.6, label='Data points')

# Fit and plot a lowess smoother for India
sns.regplot(x='SIPRI TIV for total order', y='No_Incidents', data=india_data,
            lowess=True, line_kws={'color': 'red', 'lw': 1}, label='Smoothed line')

plt.title('Impact of Arms Imports on Terrorist Incidents in India')
plt.xlabel('Total Value of Arms Imports (SIPRI TIV)')
plt.ylabel('Number of Terrorist Incidents')
plt.legend()
plt.grid(True)
plt.show()