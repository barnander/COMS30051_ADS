import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Load the data
yearly_agg_data = pd.read_csv('yearly_agg.csv')
small_arms_data = pd.read_csv('/Users/jacovaneeden/Desktop/small_arms_FINAL.csv')

# Convert 'Units' to numeric, ignoring errors to skip non-numeric values
small_arms_data['Units'] = pd.to_numeric(small_arms_data['Units'], errors='coerce')

# Filter out data before 1992
small_arms_data = small_arms_data[small_arms_data['Year'] >= 1991]
yearly_agg_data = yearly_agg_data[yearly_agg_data['iyear'] >= 1991]

# Aggregate total units imported globally per year
global_arms_imports_aggregated = small_arms_data.groupby('Year').agg(
    total_units_imported=pd.NamedAgg(column='Units', aggfunc='sum')
).reset_index()

# Rename 'Year' to 'iyear' to match the yearly_agg_data column name for merging
global_arms_imports_aggregated.rename(columns={'Year': 'iyear'}, inplace=True)

# Aggregate terror incidents and deaths data
global_terror_aggregated = yearly_agg_data.groupby('iyear').agg(
    total_incidents=pd.NamedAgg(column='No_Incidents', aggfunc='sum'),
    total_deaths=pd.NamedAgg(column='No_death', aggfunc='sum')  # Aggregating total deaths
).reset_index()

# Merge the datasets on the year
merged_global_data = pd.merge(global_arms_imports_aggregated, global_terror_aggregated, on='iyear', how='inner')

# Calculate Pearson correlation coefficient and p-values for each pair
correlation_incidents_imports, p_value_incidents_imports = pearsonr(merged_global_data['total_incidents'], merged_global_data['total_units_imported'])
correlation_deaths_imports, p_value_deaths_imports = pearsonr(merged_global_data['total_deaths'], merged_global_data['total_units_imported'])
correlation_incidents_deaths, p_value_incidents_deaths = pearsonr(merged_global_data['total_incidents'], merged_global_data['total_deaths'])

# Print the results
print("Correlation between Total Incidents and Total Units Imported: {:.2f}, P-value: {:.2e}".format(correlation_incidents_imports, p_value_incidents_imports))
print("Correlation between Total Deaths and Total Units Imported: {:.2f}, P-value: {:.2e}".format(correlation_deaths_imports, p_value_deaths_imports))
print("Correlation between Total Incidents and Total Deaths: {:.2f}, P-value: {:.2e}".format(correlation_incidents_deaths, p_value_incidents_deaths))

# Plotting the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(merged_global_data[['total_incidents', 'total_deaths', 'total_units_imported']].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, annot_kws={"size": 14})
ax.set_xticklabels(['No. of Incidents', 'No. of Deaths', 'Total Units Imported'], fontsize=12)
ax.set_yticklabels(['No. of Incidents', 'No. of Deaths', 'Total Units Imported'], fontsize=12)
plt.show()
