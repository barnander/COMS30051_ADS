import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the data with low_memory=False to handle mixed data types
yearly_agg_data = pd.read_csv('yearly_agg.csv', low_memory=False)
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv', low_memory=False)
extra_data = pd.read_csv('controls.csv')

# Convert 'Value' to numeric, handling non-numeric values by coercing them to NaN
small_arms_data['Value'] = pd.to_numeric(small_arms_data['Value'], errors='coerce')
print(small_arms_data.columns)
# Filter out data before 1992 to focus on recent data
small_arms_data = small_arms_data[small_arms_data['Year'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['Year'] >= 1991]
extra_data = extra_data[extra_data['Year'] >= 1992]

# Rename columns to align them for merging
# Using 'Reporter_Name' as 'country' and 'Year' as 'iyear' based on previous data checks
small_arms_data.rename(columns={'Reporter_Name': 'Country'}, inplace=True)
yearly_agg_data.rename(columns={'country_txt': 'Country'}, inplace=True)
print(small_arms_data.columns)
# Aggregate arms imports data per country and year
# Aggregating the 'Value' column which represents the value of arms imports
country_arms_imports_aggregated = small_arms_data.groupby(['Country', 'Year']).agg(
    Value=pd.NamedAgg(column='Value', aggfunc='sum')
).reset_index()
print(country_arms_imports_aggregated.columns)
# Aggregate terror incidents data per country and year
# Aggregating the 'No_Incidents' column which represents the number of terrorist incidents
country_terror_aggregated = yearly_agg_data.groupby(['Country', 'Year']).agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum')
).reset_index()

merged_data = pd.merge(yearly_agg_data, country_arms_imports_aggregated, on=['Country', 'Year'], how='left')
merged_data = pd.merge(merged_data, extra_data, on=['Country', 'Year'], how='left')

# Fill missing values with 0 (assuming missing data implies no orders/incidents)
merged_data.fillna(0, inplace=True)

merged_data.to_csv("Small_arms_merged", index=False)


# Prepare dependent and independent variables, include the transformed 'SIPRI TIV for total order_log'
independent_vars = merged_data[['Value', 'gdp', 'population', 'hdi', 'v2x_polyarchy']]
dependent_var = merged_data['No_Incidents']

# Standardize the independent variables including the transformed 'SIPRI TIV for total order_log'

# Add a constant for the intercept
X = sm.add_constant(independent_vars)

# Fit the OLS model with normalized variables
model_ols = sm.OLS(dependent_var, X).fit()
print(model_ols.summary())

# Plotting the relationship using the transformed and standardized 'SIPRI TIV for total order_log'
plt.figure(figsize=(10, 6))
sns.regplot(x=independent_vars['Value'], y=dependent_var, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.xlabel('Value of Arms SALW Imports')
plt.ylabel('Number of Terrorist Incidents')
plt.grid(True)
plt.show()

