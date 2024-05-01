
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the data with low_memory=False to handle mixed data types
yearly_agg_data = pd.read_csv('yearly_agg.csv', low_memory=False)
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv', low_memory=False)

# Convert 'Value' to numeric, handling non-numeric values by coercing them to NaN
small_arms_data['Value'] = pd.to_numeric(small_arms_data['Value'], errors='coerce')

# Filter out data before 1992 to focus on recent data
small_arms_data = small_arms_data[small_arms_data['Year'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['iyear'] >= 1991]

# Rename columns to align them for merging
# Using 'Reporter_Name' as 'country' and 'Year' as 'iyear' based on previous data checks
small_arms_data.rename(columns={'Reporter_Name': 'country', 'Year': 'iyear'}, inplace=True)
yearly_agg_data.rename(columns={'country_txt': 'country'}, inplace=True)

# Aggregate arms imports data per country and year
# Aggregating the 'Value' column which represents the value of arms imports
country_arms_imports_aggregated = small_arms_data.groupby(['country', 'iyear']).agg(
    total_value_imported=pd.NamedAgg(column='Value', aggfunc='sum')
).reset_index()

# Aggregate terror incidents data per country and year
# Aggregating the 'No_Incidents' column which represents the number of terrorist incidents
country_terror_aggregated = yearly_agg_data.groupby(['country', 'iyear']).agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum')
).reset_index()

# Merge the datasets on both country and year
merged_country_data = pd.merge(country_arms_imports_aggregated, country_terror_aggregated, on=['country', 'iyear'], how='inner')

# Apply log transformation with a small constant to avoid log(0)
merged_country_data['log_total_value_imported'] = np.log1p(merged_country_data['total_value_imported'])
merged_country_data['log_total_incidents'] = np.log1p(merged_country_data['total_incidents'])

# Redo the linear regression with the transformed data
X_log = merged_country_data[['log_total_value_imported']]
X_log = sm.add_constant(X_log)  # adding a constant to include an intercept in the regression model
y_log = merged_country_data['log_total_incidents']

model_log = sm.OLS(y_log, X_log).fit()  # OLS regression with log-transformed data
predictions_log = model_log.predict(X_log)  # make the predictions by the model

# Print out the regression statistics
print(model_log.summary())

# Plotting the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_total_value_imported', y='log_total_incidents', data=merged_country_data)
plt.plot(merged_country_data['log_total_value_imported'], predictions_log, color='red')  # Add the regression line
plt.title('Log-Transformed Relationship between Value of Arms Imported and Terrorist Incidents per Country per Year')
plt.xlabel('Log of Total Value of Arms Imported')
plt.ylabel('Log of Total Terrorist Incidents')
plt.grid(True)
plt.show()
