
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS

# Load the data
trade_register_data = pd.read_csv('trade-register.csv', encoding='ISO-8859-1')
yearly_agg_data = pd.read_csv('yearly_agg.csv')
extra_data = pd.read_csv('/Users/jacovaneeden/Desktop/controls.csv')

# Prepare and merge the data
arms_aggregated = trade_register_data.groupby(['Recipient', 'Year of order']).agg({
    'SIPRI TIV for total order': 'sum'
}).reset_index().rename(columns={'Recipient': 'Country', 'Year of order': 'Year'})

# Renaming columns for merging in extra_data
extra_data.rename(columns={'Country Name_x': 'Country', 'year': 'Year'}, inplace=True)

# Merge datasets
yearly_agg_data.rename(columns={'country_txt': 'Country', 'iyear': 'Year'}, inplace=True)
merged_data = pd.merge(yearly_agg_data, arms_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

# Update overlapping columns with possibly more accurate or recent data
merged_data.update(extra_data[['gdp', 'population', 'hdi', 'v2x_polyarchy']])

# Handling missing values
merged_data.fillna(merged_data.mean(numeric_only=True), inplace=True)

# Define your dependent and independent variables
dep_var = merged_data['No_Incidents']
indep_vars = merged_data[['SIPRI TIV for total order', 'gdp', 'population', 'hdi', 'v2x_polyarchy']]

# Include year and entity fixed effects
exog_vars = sm.add_constant(indep_vars)  # Adds constant term to the model
exog_vars['Year'] = pd.Categorical(merged_data['Year'])  # Treat 'Year' as fixed effects

# Using PanelOLS from linearmodels for fixed effects model
panel_data = merged_data.set_index(['Country', 'Year'])
mod = PanelOLS(dep_var, exog_vars, entity_effects=True, time_effects=True)
fe_res = mod.fit(cov_type='robust')

# Print results
print(fe_res)

# Optional: IV 2SLS approach
# You will need to define an appropriate instrument and include it in your analysis
iv_mod = IV2SLS.from_formula('No_Incidents ~ 1 + [SIPRI TIV for total order ~ Instrument] + gdp + population + hdi + v2x_polyarchy', data=merged_data)
iv_res = iv_mod.fit(cov_type='robust')

# Compare OLS with 2SLS if using IV
results_table = summary_col(results=[fe_res, iv_res], model_names=['Fixed Effects', 'IV 2SLS'], stars=True)
print(results_table)
